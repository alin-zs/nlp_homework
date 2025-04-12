# ------------------------
# 测试集评估
# ------------------------
print("\n=====================")
print("开始评估测试集性能...")
print(f"加载模型参数：{train_state['model_filename']}")

# 加载最佳模型
classifier.load_state_dict(torch.load(train_state['model_filename']))
classifier = classifier.to(args.device)
classifier.eval()

# 准备测试集
dataset.set_split('test')
batch_generator = generate_batches(
    dataset,
    batch_size=args.batch_size,
    device=args.device,
    shuffle=False
)

running_loss = 0.0
running_corrects = 0
total_samples = 0

with torch.no_grad():
    for batch_dict in batch_generator:
        x_data = batch_dict['x_data']
        y_target = batch_dict['y_target']
        x_lengths = batch_dict['x_length']

        y_pred = classifier(x_data, x_lengths)
        loss = loss_func(y_pred, y_target)

        running_loss += loss.item() * x_data.size(0)
        _, y_pred_indices = y_pred.max(dim=1)
        running_corrects += torch.eq(y_pred_indices, y_target).sum().item()
        total_samples += x_data.size(0)

test_loss = running_loss / total_samples
test_acc = running_corrects / total_samples * 100

# 输出结果
print(f"测试集损失值：{test_loss:.4f}")
print(f"测试集准确率：{test_acc:.2f}%")
print("=====================")