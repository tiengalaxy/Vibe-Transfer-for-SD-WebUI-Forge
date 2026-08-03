import os
import stat
import torch
import numpy as np
from PIL import Image
import gradio as gr
import modules.scripts as scripts
from modules import shared
from modules.processing import StableDiffusionProcessing
from collections import OrderedDict
import traceback

MAX_VIBE_SLOTS = 4

CN_MODEL_EXTS = [".pt", ".pth", ".ckpt", ".safetensors", ".bin"]

_ipadapter_filename_dict = OrderedDict()
_ipadapter_names = ['None']


def _numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    return torch.from_numpy(y).float()


def _get_webui_root():
    try:
        from modules.paths_internal import data_path
        return data_path
    except ImportError:
        pass
    try:
        from modules import paths
        return paths.data_path
    except ImportError:
        pass
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_model_dirs():
    dirs = []

    try:
        from modules.paths_internal import models_path
        dirs.append(os.path.join(models_path, 'ControlNet'))
        dirs.append(os.path.join(models_path, 'ipadapter'))
    except ImportError:
        pass

    try:
        from modules_forge.shared import controlnet_dir
        if controlnet_dir:
            dirs.append(controlnet_dir)
    except ImportError:
        pass

    try:
        extra = shared.opts.data.get("control_net_models_path", None)
        if extra and os.path.isdir(extra):
            dirs.append(extra)
    except Exception:
        pass

    root = _get_webui_root()
    fallback_dirs = [
        os.path.join(root, "models", "ControlNet"),
        os.path.join(root, "models", "ipadapter"),
        os.path.join(root, "models", "controlnet"),
    ]
    for d in fallback_dirs:
        if d not in dirs:
            dirs.append(d)

    unique = []
    seen = set()
    for d in dirs:
        nd = os.path.normpath(os.path.abspath(d))
        if nd not in seen:
            seen.add(nd)
            unique.append(nd)
    return unique


def _traverse_all_files(curr_path, model_list):
    if not os.path.isdir(curr_path):
        return model_list
    try:
        entries = list(os.scandir(curr_path))
    except OSError:
        return model_list
    for entry in entries:
        try:
            if entry.is_file(follow_symlinks=False):
                if os.path.splitext(entry.name)[1] in CN_MODEL_EXTS:
                    model_list.append(entry.path)
            elif entry.is_dir(follow_symlinks=False):
                model_list = _traverse_all_files(entry.path, model_list)
        except OSError:
            continue
    return model_list


def _is_ipadapter_file(filepath):
    name_lower = os.path.basename(filepath).lower()
    keywords = ['ip-adapter', 'ip_adapter', 'ipadapter', 'faceid', 'face_id', 'instantid', 'instant_id']
    if any(kw in name_lower for kw in keywords):
        return True

    try:
        if filepath.lower().endswith('.safetensors'):
            # Only read metadata/header keys, do not load weights
            from safetensors import safe_open
            with safe_open(filepath, framework='pt', device='cpu') as f:
                keys = list(f.keys())[:20]
            if any(k.startswith("image_proj.") or k.startswith("ip_adapter.") for k in keys):
                return True
        else:
            # Only scan known IP-Adapter file names for pickle-based formats
            # to avoid arbitrary code execution from torch.load(weights_only=False)
            print(f"[VibeTransfer] 跳过 pickle 格式文件的安全扫描: {os.path.basename(filepath)}")
    except Exception as e:
        print(f"[VibeTransfer] 检查文件失败 {os.path.basename(filepath)}: {e}")
    return False


def update_ipadapter_filenames():
    global _ipadapter_filename_dict, _ipadapter_names

    _ipadapter_filename_dict = OrderedDict()
    _ipadapter_names = ['None']

    dirs = _get_model_dirs()
    print(f"[VibeTransfer] 搜索目录: {dirs}")

    for d in dirs:
        if not os.path.isdir(d):
            continue
        print(f"[VibeTransfer] 扫描: {d}")
        files = _traverse_all_files(d, [])
        print(f"[VibeTransfer]   文件数: {len(files)}")

        for filepath in sorted(files):
            if _is_ipadapter_file(filepath):
                basename = os.path.basename(filepath)
                name_no_ext = os.path.splitext(basename)[0]
                if name_no_ext == "None":
                    continue
                try:
                    from modules import sd_models
                    h = sd_models.model_hash(filepath)
                    display = f"{name_no_ext} [{h}]"
                except Exception:
                    display = name_no_ext
                if display not in _ipadapter_filename_dict:
                    _ipadapter_filename_dict[display] = filepath
                    print(f"[VibeTransfer]   找到: {display}")

    _ipadapter_names = ['None'] + list(_ipadapter_filename_dict.keys())
    count = len(_ipadapter_names) - 1
    if count == 0:
        print("[VibeTransfer] 未找到 IP-Adapter 模型!")
    else:
        print(f"[VibeTransfer] 共找到 {count} 个 IP-Adapter 模型")
    return _ipadapter_names


def _load_ipadapter_via_forge(model_path):
    try:
        from modules_forge.shared import try_load_supported_control_model
        patcher = try_load_supported_control_model(model_path)
        if patcher is not None:
            cls_name = patcher.__class__.__name__
            if 'IPAdapter' in cls_name or 'ipadapter' in cls_name.lower():
                print(f"[VibeTransfer] 通过 Forge API 加载成功: {cls_name}")
                return patcher
            else:
                print(f"[VibeTransfer] 模型类型不是 IP-Adapter: {cls_name}")
        else:
            print("[VibeTransfer] Forge API 返回 None")
    except ImportError as e:
        print(f"[VibeTransfer] Forge API 不可用: {e}")
    except Exception as e:
        print(f"[VibeTransfer] Forge API 加载失败: {e}")
    return None


def _load_ipadapter_manual(model_path):
    try:
        if model_path.lower().endswith('.safetensors'):
            from safetensors.torch import load_file
            raw = load_file(model_path, device='cpu')
        else:
            try:
                raw = torch.load(model_path, map_location='cpu', weights_only=True)
            except Exception:
                print("[VibeTransfer] weights_only=True 加载失败，尝试非安全模式 (仅当来源可信)")
                raw = torch.load(model_path, map_location='cpu', weights_only=False)

        if not isinstance(raw, dict):
            return None

        st_model = {"image_proj": {}, "ip_adapter": {}}
        for key in raw.keys():
            if key.startswith("image_proj."):
                st_model["image_proj"][key.replace("image_proj.", "")] = raw[key]
            elif key.startswith("ip_adapter."):
                st_model["ip_adapter"][key.replace("ip_adapter.", "")] = raw[key]

        if len(st_model["ip_adapter"]) > 0:
            print(f"[VibeTransfer] 手动加载成功, ip_adapter keys: {len(st_model['ip_adapter'])}")
            return st_model
    except Exception as e:
        print(f"[VibeTransfer] 手动加载失败: {e}")
    return None


def _load_clip_vision():
    try:
        from modules_forge.shared import supported_preprocessors
        for name, p in supported_preprocessors.items():
            if hasattr(p, 'load_clipvision'):
                try:
                    cv = p.load_clipvision()
                    if cv is not None:
                        print(f"[VibeTransfer] CLIP Vision 已加载: {name}")
                        return cv
                except Exception as e:
                    print(f"[VibeTransfer] 加载 CLIP Vision 预处理器失败 ({name}): {e}")
                    continue
    except Exception as e:
        print(f"[VibeTransfer] 获取 supported_preprocessors 失败: {e}")
        traceback.print_exc()

    try:
        from modules_forge.supported_preprocessor import PreprocessorClipVision
        for name in ['CLIP-ViT-H (IPAdapter)', 'CLIP-ViT-bigG (IPAdapter)']:
            try:
                from modules_forge.shared import supported_preprocessors
                if name in supported_preprocessors:
                    cv = supported_preprocessors[name].load_clipvision()
                    if cv is not None:
                        return cv
            except Exception as e:
                print(f"[VibeTransfer] 加载 {name} 失败: {e}")
                continue
    except Exception as e:
        print(f"[VibeTransfer] 导入 PreprocessorClipVision 失败: {e}")
        traceback.print_exc()

    print("[VibeTransfer] CLIP Vision 加载失败")
    return None


def compute_information_extracted_mapping(info_extracted, is_sdxl=False):
    start_pct = 0.0
    end_pct = 1.0
    noise_level = 0.0

    if info_extracted <= 0.0:
        start_pct = 0.0
        end_pct = 0.0
        noise_level = 0.0
    elif info_extracted <= 0.3:
        end_pct = info_extracted / 0.3 * 0.5
        noise_level = 0.3 - info_extracted
    elif info_extracted <= 0.7:
        start_pct = (info_extracted - 0.3) / 0.4 * 0.2
        end_pct = 0.5 + (info_extracted - 0.3) / 0.4 * 0.4
    else:
        start_pct = 0.2 + (info_extracted - 0.7) / 0.3 * 0.5
        end_pct = 0.9 + (info_extracted - 0.7) / 0.3 * 0.1

    if is_sdxl:
        end_pct = min(end_pct, 0.85)

    return start_pct, end_pct, noise_level


def _get_attention_function():
    try:
        from backend import attention
        return attention.attention_function
    except ImportError:
        pass
    try:
        from ldm_patched.modules import attention
        return attention.attention_function
    except ImportError:
        pass

    def fallback_attention(q, k, v, heads, mask=None):
        b, _, dim_head = q.shape
        q = q.reshape(b, -1, heads, dim_head // heads).transpose(1, 2)
        k = k.reshape(b, -1, heads, dim_head // heads).transpose(1, 2)
        v = v.reshape(b, -1, heads, dim_head // heads).transpose(1, 2)
        # scaled_dot_product_attention applies scale internally
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return out.transpose(1, 2).reshape(b, -1, dim_head)

    return fallback_attention


def _get_sampling_function():
    try:
        from backend.sampling.sampling_function import sampling_function_inner
        return sampling_function_inner
    except ImportError:
        pass
    try:
        from ldm_patched.modules.samplers import sampling_function
        return sampling_function
    except ImportError:
        pass
    return None


def apply_reference_style_transfer(process, ref_pil, weight, style_fidelity, start_pct, end_pct):
    attn_fn = _get_attention_function()
    sampling_fn = _get_sampling_function()

    if sampling_fn is None:
        print("[VibeTransfer] 采样函数不可用，尝试直接注入注意力钩子")

    try:
        vae = process.sd_model.forge_objects.vae
        ref_arr = np.array(ref_pil.convert("RGB")).astype(np.float32) / 255.0
        ref_tensor = torch.from_numpy(ref_arr).unsqueeze(0)

        if hasattr(vae, 'device'):
            ref_tensor = ref_tensor.to(vae.device)

        latent_image = vae.encode(ref_tensor.movedim(1, -1))
        latent_image = process.sd_model.forge_objects.vae.first_stage_model.process_in(latent_image)

        gen_seed = process.seeds[0] + 1 if process.seeds else 0
        gen_cpu = torch.Generator().manual_seed(gen_seed)

        unet = process.sd_model.forge_objects.unet.clone()

        sigma_max = unet.model.predictor.percent_to_sigma(start_pct)
        sigma_min = unet.model.predictor.percent_to_sigma(end_pct)

        recorded_attn1 = {}
        recorded_h = {}
        is_recording = [False]

        def sdp(q, k, v, transformer_options):
            if q.shape[0] == 0:
                return q
            return attn_fn(q, k, v, heads=transformer_options["n_heads"], mask=None)

        def adain(x, target_std, target_mean):
            if x.shape[0] == 0:
                return x
            std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True, correction=0)
            return (((x - mean) / std) * target_std) + target_mean

        def zero_cat(a, b, dim):
            if a.shape[0] == 0:
                return b
            if b.shape[0] == 0:
                return a
            return torch.cat([a, b], dim=dim)

        if sampling_fn is not None:
            def conditioning_modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
                sigma = timestep[0].item()
                if not (sigma_min <= sigma <= sigma_max):
                    return model, x, timestep, uncond, cond, cond_scale, model_options, seed
                is_recording[0] = True
                try:
                    xt = latent_image.to(x) + torch.randn(x.size(), dtype=x.dtype, generator=gen_cpu).to(x) * sigma
                    sampling_fn(model, xt, timestep, uncond, cond, 1, model_options, seed)
                except Exception as e:
                    print(f"[VibeTransfer] Reference 采样步骤失败: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    is_recording[0] = False
                return model, x, timestep, uncond, cond, cond_scale, model_options, seed

            unet.add_conditioning_modifier(conditioning_modifier)

        def block_proc(h, flag, transformer_options):
            if flag != 'after':
                return h
            location = transformer_options['block']
            sigma = transformer_options["sigmas"][0].item()
            if not (sigma_min <= sigma <= sigma_max):
                return h
            channel = int(h.shape[1])
            minimal_channel = 1500 - 1000 * weight
            if channel < minimal_channel:
                return h
            if is_recording[0]:
                recorded_h[location] = torch.std_mean(h, dim=(2, 3), keepdim=True, correction=0)
                return h
            else:
                if location not in recorded_h:
                    return h
                cond_indices = transformer_options['cond_indices']
                uncond_indices = transformer_options['uncond_indices']
                cond_or_uncond = transformer_options['cond_or_uncond']
                r_std, r_mean = recorded_h[location]
                h_c = h[cond_indices]
                h_uc = h[uncond_indices]
                o_c = adain(h_c, r_std, r_mean)
                o_uc_strong = h_uc
                o_uc_weak = adain(h_uc, r_std, r_mean)
                o_uc = o_uc_weak + (o_uc_strong - o_uc_weak) * style_fidelity
                recon = []
                for cx in cond_or_uncond:
                    if cx == 0:
                        recon.append(o_c)
                    else:
                        recon.append(o_uc)
                o = torch.cat(recon, dim=0)
                return o

        def attn1_proc(q, k, v, transformer_options):
            sigma = transformer_options["sigmas"][0].item()
            if not (sigma_min <= sigma <= sigma_max):
                return sdp(q, k, v, transformer_options)
            location = (transformer_options['block'][0], transformer_options['block'][1],
                        transformer_options['block_index'])
            channel = int(q.shape[2])
            minimal_channel = 1500 - 1280 * weight
            if channel < minimal_channel:
                return sdp(q, k, v, transformer_options)
            if is_recording[0]:
                recorded_attn1[location] = (k, v)
                return sdp(q, k, v, transformer_options)
            else:
                if location not in recorded_attn1:
                    return sdp(q, k, v, transformer_options)
                cond_indices = transformer_options['cond_indices']
                uncond_indices = transformer_options['uncond_indices']
                cond_or_uncond = transformer_options['cond_or_uncond']
                q_c = q[cond_indices]
                q_uc = q[uncond_indices]
                k_c = k[cond_indices]
                k_uc = k[uncond_indices]
                v_c = v[cond_indices]
                v_uc = v[uncond_indices]
                k_r, v_r = recorded_attn1[location]
                o_c = sdp(q_c, zero_cat(k_c, k_r, dim=1), zero_cat(v_c, v_r, dim=1), transformer_options)
                o_uc_strong = sdp(q_uc, k_uc, v_uc, transformer_options)
                o_uc_weak = sdp(q_uc, zero_cat(k_uc, k_r, dim=1), zero_cat(v_uc, v_r, dim=1), transformer_options)
                o_uc = o_uc_weak + (o_uc_strong - o_uc_weak) * style_fidelity
                recon = []
                for cx in cond_or_uncond:
                    if cx == 0:
                        recon.append(o_c)
                    else:
                        recon.append(o_uc)
                o = torch.cat(recon, dim=0)
                return o

        unet.add_block_modifier(block_proc)
        unet.set_model_replace_all(attn1_proc, 'attn1')
        process.sd_model.forge_objects.unet = unet
        return True

    except Exception as e:
        print(f"[VibeTransfer] Reference 风格迁移失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def reinhard_color_transfer(source_img, target_img, weight):
    try:
        src_arr = np.array(source_img.convert("RGB")).astype(np.float32)
        tgt_arr = np.array(target_img.convert("RGB")).astype(np.float32)

        def rgb_to_lab(arr):
            arr = np.clip(arr, 0.0, 255.0) / 255.0
            l = 0.3811 * arr[..., 0] + 0.5783 * arr[..., 1] + 0.0402 * arr[..., 2]
            m = 0.1967 * arr[..., 0] + 0.7244 * arr[..., 1] + 0.0782 * arr[..., 2]
            s = 0.0241 * arr[..., 0] + 0.1288 * arr[..., 1] + 0.8444 * arr[..., 2]
            l = np.log10(np.clip(l, 1e-5, 1.0))
            m = np.log10(np.clip(m, 1e-5, 1.0))
            s = np.log10(np.clip(s, 1e-5, 1.0))
            L = (l + m + s) / np.sqrt(3)
            A = (l + m - 2 * s) / np.sqrt(6)
            B = (l - m) / np.sqrt(2)
            return np.stack([L, A, B], axis=-1)

        def lab_to_rgb(arr):
            L, A, B = arr[..., 0], arr[..., 1], arr[..., 2]
            l = L / np.sqrt(3) + A / np.sqrt(6) + B / np.sqrt(2)
            m = L / np.sqrt(3) + A / np.sqrt(6) - B / np.sqrt(2)
            s = L / np.sqrt(3) - 2 * A / np.sqrt(6)
            l = np.power(10.0, np.clip(l, -5.0, 0.0))
            m = np.power(10.0, np.clip(m, -5.0, 0.0))
            s = np.power(10.0, np.clip(s, -5.0, 0.0))
            R = 4.4679 * l - 3.5873 * m + 0.1193 * s
            G = -1.2186 * l + 2.3809 * m - 0.1624 * s
            B_coeff = 0.0497 * l - 0.2439 * m + 1.2045 * s
            RGB = np.stack([R, G, B_coeff], axis=-1)
            return np.clip(RGB * 255.0, 0, 255).astype(np.uint8)

        src_lab = rgb_to_lab(src_arr)
        tgt_lab = rgb_to_lab(tgt_arr)
        src_mean = np.mean(src_lab, axis=(0, 1))
        tgt_mean = np.mean(tgt_lab, axis=(0, 1))
        src_std = np.std(src_lab, axis=(0, 1))
        tgt_std = np.std(tgt_lab, axis=(0, 1))
        src_std = np.where(src_std < 1e-4, 1e-4, src_std)
        scaled_lab = (src_lab - src_mean) * (tgt_std / src_std) + tgt_mean
        blended_lab = src_lab + (scaled_lab - src_lab) * weight
        transfer_rgb = lab_to_rgb(blended_lab)
        return Image.fromarray(transfer_rgb)
    except Exception as e:
        print(f"[VibeTransfer] Reinhard 色彩迁移失败: {e}")
        return source_img


MODE_IPADAPTER = "IP-Adapter 语义氛围 (Semantic Vibe)"
MODE_REFERENCE = "Reference 注意力+AdaIN (Attention Style)"
MODE_HYBRID = "混合模式 (Hybrid)"
MODE_COLOR = "仅色彩光影 (Color/Lighting Only)"


class VibeTransferForgeScript(scripts.Script):

    def title(self):
        return "Vibe Transfer (氛围转移)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        vibe_components = []

        update_ipadapter_filenames()

        with gr.Accordion("Vibe Transfer (氛围转移)", open=False, elem_id="vibe_transfer_accordion"):
            with gr.Row():
                enabled = gr.Checkbox(
                    label="启用 Vibe Transfer (Enable)",
                    value=False,
                    elem_id="vibe_transfer_enabled"
                )
                transfer_mode = gr.Dropdown(
                    label="转移模式 (Transfer Mode)",
                    choices=[MODE_IPADAPTER, MODE_REFERENCE, MODE_HYBRID, MODE_COLOR],
                    value=MODE_IPADAPTER,
                    elem_id="vibe_transfer_mode"
                )

            with gr.Row():
                ipadapter_model = gr.Dropdown(
                    label="IP-Adapter 模型",
                    choices=_ipadapter_names,
                    value=_ipadapter_names[0],
                    elem_id="vibe_transfer_ipadapter_model"
                )
                refresh_btn = gr.Button("🔄 刷新模型", elem_id="vibe_transfer_refresh_btn")

                def refresh_models():
                    update_ipadapter_filenames()
                    return gr.update(choices=_ipadapter_names, value=_ipadapter_names[0] if _ipadapter_names else None)

                refresh_btn.click(
                    fn=refresh_models,
                    inputs=[],
                    outputs=[ipadapter_model]
                )

            gr.HTML(value="""
                <div style="font-size: 0.85em; color: var(--body-text-color-subdued, #666); margin-top: 4px; line-height: 1.4;">
                    <strong>模式说明：</strong><br>
                    • <strong>IP-Adapter 语义氛围</strong>：使用 CLIP Vision 提取参考图语义，通过解耦交叉注意力注入 — 最接近 NovelAI Vibe Transfer<br>
                    • <strong>Reference 注意力+AdaIN</strong>：参考图潜空间注意力替换 + 自适应实例归一化<br>
                    • <strong>混合模式</strong>：同时使用 IP-Adapter 和 Reference<br>
                    • <strong>仅色彩光影</strong>：像素级 Reinhard 色彩迁移，不改变结构
                </div>
            """)

            for slot_idx in range(MAX_VIBE_SLOTS):
                with gr.Accordion(f"Vibe #{slot_idx + 1}", open=(slot_idx == 0), elem_id=f"vibe_slot_{slot_idx}_accordion"):
                    with gr.Row():
                        slot_enabled = gr.Checkbox(
                            label=f"启用 Vibe #{slot_idx + 1}",
                            value=(slot_idx == 0),
                            elem_id=f"vibe_slot_{slot_idx}_enabled"
                        )
                    with gr.Row():
                        ref_img = gr.Image(
                            label=f"参考氛围图 #{slot_idx + 1}",
                            type="pil",
                            elem_id=f"vibe_slot_{slot_idx}_image"
                        )
                    with gr.Row():
                        ref_strength = gr.Slider(
                            minimum=0.0, maximum=2.0, step=0.05, value=0.6,
                            label=f"参考强度 Reference Strength",
                            elem_id=f"vibe_slot_{slot_idx}_ref_strength"
                        )
                        info_extracted = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.01, value=1.0,
                            label=f"信息提取量 Information Extracted",
                            elem_id=f"vibe_slot_{slot_idx}_info_extracted"
                        )
                    with gr.Row():
                        style_fidelity = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.01, value=0.5,
                            label=f"风格保真度 Style Fidelity",
                            elem_id=f"vibe_slot_{slot_idx}_style_fidelity"
                        )

                    vibe_components.append({
                        'enabled': slot_enabled,
                        'image': ref_img,
                        'ref_strength': ref_strength,
                        'info_extracted': info_extracted,
                        'style_fidelity': style_fidelity,
                    })

            gr.HTML(value="""
                <div style="font-size: 0.82em; color: var(--body-text-color-subdued, #666); margin-top: 4px; line-height: 1.4;">
                    <strong>参数说明：</strong><br>
                    • <strong>参考强度</strong>：氛围转移强度，越高参考图影响越大 — 对应 NovelAI Reference Strength<br>
                    • <strong>信息提取量</strong>：从参考图提取多少信息 — 对应 NovelAI Information Extracted<br>
                    • <strong>风格保真度</strong>：Reference/混合模式下控制无条件区域风格保留
                </div>
            """)

        all_components = [enabled, transfer_mode, ipadapter_model]
        for vc in vibe_components:
            all_components.extend([vc['enabled'], vc['image'], vc['ref_strength'], vc['info_extracted'], vc['style_fidelity']])

        return all_components

    def _parse_args(self, args):
        enabled = args[0]
        transfer_mode = args[1]
        ipadapter_model_name = args[2]

        active_vibes = []
        for i in range(MAX_VIBE_SLOTS):
            base = 3 + i * 5
            if base + 4 >= len(args):
                break
            slot_enabled = args[base]
            slot_image = args[base + 1]
            ref_strength = args[base + 2]
            info_extracted = args[base + 3]
            style_fidelity = args[base + 4]

            if slot_enabled and slot_image is not None:
                active_vibes.append({
                    'index': i,
                    'image': slot_image,
                    'ref_strength': ref_strength,
                    'info_extracted': info_extracted,
                    'style_fidelity': style_fidelity,
                })

        return enabled, transfer_mode, ipadapter_model_name, active_vibes

    def process(self, p: StableDiffusionProcessing, *args):
        enabled, transfer_mode, ipadapter_model_name, active_vibes = self._parse_args(args)

        if not enabled or not active_vibes:
            return

        is_sdxl = getattr(p.sd_model, 'is_sdxl', False) if hasattr(p, 'sd_model') and p.sd_model else False

        print(f"\n[VibeTransfer] 激活")
        print(f"  模式: {transfer_mode}, Vibe数: {len(active_vibes)}, SDXL: {is_sdxl}")

        p.extra_generation_params["Vibe Transfer Mode"] = transfer_mode
        p.extra_generation_params["Vibe Transfer Count"] = len(active_vibes)

        use_ipadapter = transfer_mode in [MODE_IPADAPTER, MODE_HYBRID]
        use_reference = transfer_mode in [MODE_REFERENCE, MODE_HYBRID]

        if use_ipadapter:
            ipadapter_ok = self._apply_ipadapter(p, active_vibes, ipadapter_model_name, is_sdxl)
            if not ipadapter_ok and not use_reference:
                print("[VibeTransfer] IP-Adapter 不可用，自动回退到 Reference 模式")
                use_reference = True

        if use_reference:
            primary_vibe = active_vibes[0]
            start_pct, end_pct, _ = compute_information_extracted_mapping(
                primary_vibe['info_extracted'], is_sdxl
            )
            success = apply_reference_style_transfer(
                process=p,
                ref_pil=primary_vibe['image'],
                weight=primary_vibe['ref_strength'],
                style_fidelity=primary_vibe['style_fidelity'],
                start_pct=start_pct,
                end_pct=end_pct
            )
            if success:
                print(f"[VibeTransfer] Reference 风格迁移已注入")

        self._active_vibes = active_vibes
        self._transfer_mode = transfer_mode

    def _apply_ipadapter(self, p, active_vibes, ipadapter_model_name, is_sdxl):
        if ipadapter_model_name == "None" or not ipadapter_model_name:
            print("[VibeTransfer] 未选择 IP-Adapter 模型")
            return False

        model_path = _ipadapter_filename_dict.get(ipadapter_model_name)
        if model_path is None:
            print(f"[VibeTransfer] 未找到模型路径: {ipadapter_model_name}")
            return False

        print(f"[VibeTransfer] 加载 IP-Adapter: {model_path}")

        ipadapter_patcher = _load_ipadapter_via_forge(model_path)

        if ipadapter_patcher is not None:
            return self._apply_ipadapter_with_patcher(p, active_vibes, ipadapter_patcher, is_sdxl)

        print("[VibeTransfer] Forge API 加载失败，尝试手动加载...")
        ipadapter_sd = _load_ipadapter_manual(model_path)
        if ipadapter_sd is not None:
            return self._apply_ipadapter_manual(p, active_vibes, ipadapter_sd, is_sdxl)

        print("[VibeTransfer] 所有加载方式均失败")
        return False

    def _apply_ipadapter_with_patcher(self, p, active_vibes, patcher, is_sdxl):
        clip_vision = _load_clip_vision()
        if clip_vision is None:
            print("[VibeTransfer] CLIP Vision 加载失败")
            return False

        for vibe in active_vibes:
            start_pct, end_pct, noise_level = compute_information_extracted_mapping(
                vibe['info_extracted'], is_sdxl
            )

            try:
                image_tensor = _numpy_to_pytorch(np.array(vibe['image'].convert("RGB")))

                cond_dict = dict(
                    clip_vision=clip_vision,
                    image=image_tensor,
                    weight_type="original",
                    noise=noise_level,
                    embeds=None,
                    unfold_batch=False,
                )

                patcher.strength = vibe['ref_strength']
                patcher.start_percent = start_pct
                patcher.end_percent = end_pct

                try:
                    from lib_ipadapter.IPAdapterPlus import IPAdapterApply
                    opApply = IPAdapterApply().apply_ipadapter
                except ImportError:
                    print("[VibeTransfer] IPAdapterApply 不可用")
                    return False

                unet = p.sd_model.forge_objects.unet
                unet = opApply(
                    ipadapter=patcher.ip_adapter,
                    model=unet,
                    weight=vibe['ref_strength'],
                    start_at=start_pct,
                    end_at=end_pct,
                    faceid_v2=getattr(patcher, 'faceid_v2', False),
                    weight_v2=getattr(patcher, 'weight_v2', False),
                    attn_mask=None,
                    **cond_dict
                )[0]
                p.sd_model.forge_objects.unet = unet

                print(f"[VibeTransfer] Vibe #{vibe['index'] + 1} 已注入 (Forge Patcher)")
            except Exception as e:
                print(f"[VibeTransfer] Vibe #{vibe['index'] + 1} 注入失败: {e}")
                import traceback
                traceback.print_exc()

        return True

    def _apply_ipadapter_manual(self, p, active_vibes, ipadapter_sd, is_sdxl):
        clip_vision = _load_clip_vision()
        if clip_vision is None:
            print("[VibeTransfer] CLIP Vision 加载失败")
            return False

        try:
            from lib_ipadapter.IPAdapterPlus import IPAdapterApply
            opApply = IPAdapterApply().apply_ipadapter
        except ImportError:
            print("[VibeTransfer] IPAdapterApply 不可用，无法手动注入")
            return False

        for vibe in active_vibes:
            start_pct, end_pct, noise_level = compute_information_extracted_mapping(
                vibe['info_extracted'], is_sdxl
            )

            try:
                image_tensor = _numpy_to_pytorch(np.array(vibe['image'].convert("RGB")))

                unet = p.sd_model.forge_objects.unet
                unet = opApply(
                    ipadapter=ipadapter_sd,
                    model=unet,
                    weight=vibe['ref_strength'],
                    start_at=start_pct,
                    end_at=end_pct,
                    faceid_v2=False,
                    weight_v2=False,
                    attn_mask=None,
                    clip_vision=clip_vision,
                    image=image_tensor,
                    weight_type="original",
                    noise=noise_level,
                    embeds=None,
                    unfold_batch=False,
                )[0]
                p.sd_model.forge_objects.unet = unet

                print(f"[VibeTransfer] Vibe #{vibe['index'] + 1} 已注入 (手动加载)")
            except Exception as e:
                print(f"[VibeTransfer] Vibe #{vibe['index'] + 1} 注入失败: {e}")
                import traceback
                traceback.print_exc()

        return True

    def postprocess_image(self, p, pp, *args):
        enabled, transfer_mode, _, active_vibes = self._parse_args(args)

        if not enabled or transfer_mode != MODE_COLOR:
            return

        if not active_vibes:
            return

        primary_vibe = active_vibes[0]
        if primary_vibe['image'] is None:
            return

        weight = primary_vibe['ref_strength']
        if weight <= 0:
            return

        try:
            transferred = reinhard_color_transfer(pp.image, primary_vibe['image'], weight)
            pp.image = transferred
            print("[VibeTransfer] 色彩迁移完成")
        except Exception as e:
            print(f"[VibeTransfer] 色彩迁移失败: {e}")

    def postprocess(self, p, processed, *args):
        if hasattr(self, '_active_vibes'):
            del self._active_vibes
        if hasattr(self, '_transfer_mode'):
            del self._transfer_mode
