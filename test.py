import torch

# quant = torch.load("./quantized_export/split_model_quantized_8bit.pt")
# for k, v in list(quant["client"].items())[:5]:
#     print(k, v["act_min"], v["act_max"])
# ====== 1. 加载模型 ======
model_path = "quantized_export/split_model_quantized_8bit.pt"
quant_dict = torch.load(model_path, map_location="cpu")

# ====== 2. 提取客户端层 ======
client_layers = quant_dict.get("client", {})
print(f"\n✅ 共 {len(client_layers)} 个客户端层：")
print(list(client_layers.keys()))  # 打印层名

# ====== 3. 查看详细参数 ======
num_layers_to_show = len(client_layers)
print("\n📊 === 客户端详细参数 ===")

for i, (lname, linfo) in enumerate(client_layers.items()):
    if i >= num_layers_to_show:
        break
    print(f"\n=== Layer {i+1}: {lname} ===")
    for k, v in linfo.items():
        # 打印标量参数
        if isinstance(v, (int, float)):
            print(f"{k:15s}: {v}")
        # 打印张量的形状和数据类型
        elif isinstance(v, torch.Tensor):
            print(f"{k:15s}: Tensor[{tuple(v.shape)}], dtype={v.dtype}")
        else:
            print(f"{k:15s}: {v}")

    # ====== 4. 打印量化权重 tensor 的部分内容 ======
    if "w_q" in linfo and isinstance(linfo["w_q"], torch.Tensor):
        w_q = linfo["w_q"]
        print("\n-- w_q 前5行前8列示例 --")
        rows = min(5, w_q.shape[0])
        cols = min(8, w_q.shape[1])
        print(w_q[:rows, :cols])
    else:
        print("\n-- 本层没有权重量化张量 (w_q) --")

print("\n✅ 打印完毕！")
