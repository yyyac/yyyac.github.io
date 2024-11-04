# pytorch 操作注意事项

## `view` 和 `reshape`

`view` 和 `reshape` 是 PyTorch 中常用的张量变形方法，两者看似相似，但在内存处理和使用限制上有一些区别。下面解释它们的区别并通过例子展示用法。

### `view`

- **作用**：`view` 是对张量进行重新形状调整，返回一个新的张量，但它共享原始张量的内存。
- **要求**：由于 `view` 共享内存，它要求原张量在内存中是连续的。使用 `view` 之前，可以调用 `.contiguous()` 方法确保连续性。
- **优点**：不会创建新的内存空间，适用于需要高效存储的情况。

**示例**：

```python
import torch

# 创建一个张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 使用 view 改变形状
tensor_view = tensor.view(3, 2)

print("Original tensor:\n", tensor)
print("View tensor:\n", tensor_view)

# 修改 view 后原始张量也会改变
tensor_view[0, 0] = 10
print("\nAfter modifying view:")
print("Original tensor:\n", tensor)
print("View tensor:\n", tensor_view)
```

### `reshape`

- **作用**：`reshape` 也用于改变张量的形状，但与 `view` 不同，它会尝试创建一个新的内存空间，且不要求输入张量在内存中是连续的。
- **灵活性**：`reshape` 会根据需要在非连续内存的情况下创建一个新的张量。若原始张量是连续的，`reshape` 通常会直接返回共享内存的新形状张量（与 `view` 类似）；否则会创建一个新的副本。
- **适用场景**：对于内存非连续的张量，可以直接使用 `reshape` 而不需要手动调用 `.contiguous()`。

**示例**：

```python
import torch

# 创建一个张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 使用 reshape 改变形状
tensor_reshape = tensor.reshape(3, 2)

print("Original tensor:\n", tensor)
print("Reshape tensor:\n", tensor_reshape)

# 修改 reshape 后原始张量也会改变（若没有创建新的内存空间）
tensor_reshape[0, 0] = 20
print("\nAfter modifying reshape:")
print("Original tensor:\n", tensor)
print("Reshape tensor:\n", tensor_reshape)
```