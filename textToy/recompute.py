# -*- coding: UTF-8 -*-
"""
tensorflow静态图重计算方法，改自 tensorflow1.15 的接口
@author: huanghui
@file: recompute.py
@date: 2020/10/07
"""
from tensorflow.python.ops import variable_scope
from tensorflow.python.eager import backprop
from tensorflow.python.framework import ops
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.eager import tape as tape_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops.custom_gradient import copy_handle_data
from tensorflow.python.util import tf_decorator


def _graph_mode_decorator(f, *args, **kwargs):
    """Implement custom gradient decorator for graph mode."""
    # 原始tensorflow内置接口不支持kwargs，应该是怕出现问题，这里我把kwargs加进来
    # 但需要注意的是args传入的值tensor，kwargs传入的是f的固定参数，一定不能搞错

    name = "CustomGradient-%s" % ops.uid()
    args = [ops.convert_to_tensor(x) for x in args]

    # Checking global and local variables attempts to ensure that no non-resource
    # Variables are added to the graph.
    current_var_scope = variable_scope.get_variable_scope()
    before_vars = set(current_var_scope.global_variables() +
                      current_var_scope.local_variables())
    with backprop.GradientTape() as tape:
        result, grad_fn = f(*args, **kwargs)

    after_vars = set(current_var_scope.global_variables() +
                     current_var_scope.local_variables())

    new_vars = after_vars - before_vars

    for v in new_vars:
        if not resource_variable_ops.is_resource_variable(v):
            raise TypeError(
                "All variables used by a function wrapped with @custom_gradient must "
                "be `ResourceVariable`s. Ensure that no `variable_scope` is created "
                "with `use_resource=False`.")
    # The variables that grad_fn needs to return gradients for are the set of
    # variables used that are *not* part of the inputs.
    variables = list(set(tape.watched_variables()) - set(args))
    grad_argspec = tf_inspect.getfullargspec(grad_fn)
    variables_in_signature = ("variables" in grad_argspec.args or
                              grad_argspec.varkw)
    if variables and not variables_in_signature:
        raise TypeError("If using @custom_gradient with a function that "
                        "uses variables, then grad_fn must accept a keyword "
                        "argument 'variables'.")
    if variables_in_signature and not variables:
        # User seems to intend to use variables but none were captured.
        if not variable_scope.get_variable_scope().use_resource:
            raise TypeError("If using @custom_gradient with a function that "
                            "uses variables, the enclosing variable scope must "
                            "have use_resource=True.")
        else:
            logging.warn("@custom_gradient grad_fn has 'variables' in signature, but "
                         "no ResourceVariables were used on the forward pass.")
    flat_result = nest.flatten(result)
    all_tensors = flat_result + args + variables

    def tape_grad_fn(*result_grads):
        """Custom grad fn wrapper."""
        result_grads = result_grads[:len(flat_result)]
        if variables:
            input_grads, variable_grads = grad_fn(*result_grads, variables=variables)
            if len(variable_grads) != len(variables):
                raise ValueError("Must return gradient for each variable from "
                                 "@custom_gradient grad_fn.")
        else:
            input_grads = grad_fn(*result_grads)
            variable_grads = []

        # Need to return one value per input to the IdentityN, so pad the
        # gradients of the inputs of the custom_gradient function with the
        # gradients of the outputs as well.
        input_grads = nest.flatten(input_grads)
        return ([None] * len(flat_result)) + input_grads + variable_grads

    @ops.RegisterGradient(name)
    def internal_grad_fn(unused_op, *result_grads):  # pylint: disable=unused-variable
        """Custom grad fn wrapper."""
        return tape_grad_fn(*result_grads)

    original_tensors = all_tensors
    with ops.get_default_graph().gradient_override_map({"IdentityN": name}):
        all_tensors = array_ops.identity_n(all_tensors)

    original_tensors = [ops.convert_to_tensor(x) for x in original_tensors]

    # Propagate handle data for happier shape inference for resource variables.
    for i, t in enumerate(original_tensors):
        if t.dtype == dtypes.resource and hasattr(t, "_handle_data"):
            all_tensors[i]._handle_data = t._handle_data  # pylint: disable=protected-access
    tape_lib.record_operation(
        f.__name__, all_tensors, original_tensors, tape_grad_fn)
    for ot, t in zip(original_tensors, all_tensors):
        copy_handle_data(ot, t)
    return nest.pack_sequence_as(
        structure=result, flat_sequence=all_tensors[:len(flat_result)])


def custom_gradient(f):
    def decorated(*args, **kwargs):
        """Decorated function with custom gradient."""
        return _graph_mode_decorator(f, *args, **kwargs)

    return tf_decorator.make_decorator(f, decorated)


def recompute_grad(f):
    """An eager-compatible version of recompute_grad.
    For f(*args, **kwargs), this supports gradients with respect to args, or to
    gradients with respect to any variables residing in the kwarg 'variables'.
    Note that for keras layer and model objects, this is handled automatically.
    Warning: If `f` was originally a tf.keras Model or Layer object, `g` will not
    be able to access the member variables of that object, because `g` returns
    through the wrapper function `inner`.  When recomputing gradients through
    objects that inherit from keras, we suggest keeping a reference to the
    underlying object around for the purpose of accessing these variables.
    Args:
      f: function `f(*x)` that returns a `Tensor` or sequence of `Tensor` outputs.
    Returns:
     A function `g` that wraps `f`, but which recomputes `f` on the backwards
     pass of a gradient call.
    """

    @custom_gradient
    def inner(*args, **kwargs):
        """Inner function closure for calculating gradients."""
        result = f(*args, **kwargs)

        def grad(dresult, variables=None):
            """Gradient function calculation for inner function."""
            with backprop.GradientTape() as t:
                t.watch(args)
                if variables is not None:
                    t.watch(variables)
                with ops.control_dependencies([dresult]):
                    result = f(*args, **kwargs)
            kw_vars = []
            if variables is not None:
                kw_vars = list(variables)
            grads = t.gradient(
                result, list(args) + kw_vars, output_gradients=[dresult])
            return grads[:len(args)], grads[len(args):]

        return result, grad

    return inner
