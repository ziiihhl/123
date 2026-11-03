# 可直接使用的模型提示词模板

2026-11-04。固定前缀与槽位会作为系统模板发送给提示词助手。大括号是待填写字段，实际返回必须替换或移除；作者依据与本服务组合分别标注。

Turbo 在 CFG=1 时默认返回空负面；其他具有真实负面输入且 CFG>1 的工作流，生成新提示词时应填入对应负面模板。用户明确要求优先于默认前缀。

## MiaoMiao Harem Anima 2.9B Beta 1.1

### 正向结构

```text
masterpiece, best quality, score_7, score_9, very aesthetic, ultra detailed, {count}, {character_and_series}, {appearance}, {clothes}, {pose_and_expression}, {action_and_spatial_relations}, {setting}, {composition}, {lighting}, {style} . {scene_sentences}
```

### 负向结构

```text
worst quality, low quality, score_1, score_2, score_3, watermark, signature, unwanted lettering, {scene_exclusions}
```

### 填写示例

```text
masterpiece, best quality, score_7, score_9, very aesthetic, ultra detailed, 1woman, solo, original adult character, short brown hair, green eyes, green cardigan, white shirt, seated, relaxed smile, holding a blue ceramic cup with both hands, window-side cafe table, potted plants, rainy street outside, waist-up view, eye-level angle, soft overcast window light, anime illustration, clean linework. An adult woman sits at a cafe table beside a rain-streaked window and holds a blue cup with both hands. Potted plants frame the window, with a wet street softly visible beyond it.
```

负向示例：`worst quality, low quality, score_1, score_2, score_3, watermark, signature, unwanted lettering, fused fingers`

Use this quality prefix by default. fair skin, sensitive and high contrast from the author example are conditional attributes, not automatic additions. Put concrete background objects and interaction geometry in the filled slots.

依据：Author quality prefix retained; scene slots and edited negative are service composition, not a verbatim author recipe.

[来源 1](https://civitai.com/models/934764?modelVersionId=3360028)；[来源 2](https://civitai.com/api/v1/model-versions/3360028)

## Hassaku XL Illustrious v3.4

### 正向结构

```text
{count}, {character_and_series}, {appearance}, {clothes}, {pose_and_expression}, {action_and_spatial_relations}, {setting}, {composition}, {lighting}, {style}
```

### 负向结构

```text
signature, watermark, blurred outlines, malformed anatomy, {scene_exclusions}
```

### 填写示例

```text
1woman, solo, original adult character, short brown hair, green eyes, green cardigan, white shirt, seated, relaxed smile, holding a blue ceramic cup with both hands, window-side cafe table, potted plants, rainy street outside, waist-up view, eye-level angle, soft overcast window light, anime illustration, clean linework
```

负向示例：`signature, watermark, blurred outlines, malformed hands`

Start with people count, then character tags; omit character slots for objects. Do not insert highres, score_* or separate franchise tags. State hair and eye colors separately; use concrete pose and framing tags.

依据：Author tag ordering and signature exclusion; the remaining slots/default negatives are service composition.

[来源 1](https://civitai.com/models/140272?modelVersionId=2615702)；[来源 2](https://civitai.com/api/v1/models/140272)

## One Obsession v2.4 Illustrious

### 正向结构

```text
masterpiece, best quality, amazing quality, {count}, {character_and_series}, {appearance}, {clothes}, {pose_and_expression}, {action_and_spatial_relations}, {setting}, {composition}, {lighting}, {style} . {scene_sentences}
```

### 负向结构

```text
worst quality, low quality, bad anatomy, watermark, logo, signature, unwanted lettering, {scene_exclusions}
```

### 填写示例

```text
masterpiece, best quality, amazing quality, 1woman, solo, original adult character, short brown hair, green eyes, green cardigan, white shirt, seated, relaxed smile, holding a blue ceramic cup with both hands, window-side cafe table, potted plants, rainy street outside, waist-up view, eye-level angle, soft overcast window light, anime illustration, clean linework
```

负向示例：`worst quality, low quality, bad anatomy, watermark, logo, signature, unwanted lettering`

Use concise tags; do not paste every parent-page quality tag. For an explicitly flat 2D request use anime coloring, flat colors; use dimensional shading only when requested. Do not ban backlighting or a simple background unless the scene requires it.

依据：Short prefix taken from the broader author guide; slot order, trimmed negative and example are service defaults, not a tested v2.4 optimum.

[来源 1](https://civitai.com/models/1318945?modelVersionId=3218603)；[来源 2](https://civitai.com/api/v1/models/1318945)

## Anima Turbo v1.1

### 正向结构

```text
masterpiece, best quality, {count}, {character_and_series}, {appearance}, {clothes}, {pose_and_expression}, {action_and_spatial_relations}, {setting}, {composition}, {lighting}, {style} . {scene_sentences}
```

### 负向结构

```text
blurred edges, malformed anatomy, unwanted lettering, watermark, {scene_exclusions}
```

### 填写示例

```text
masterpiece, best quality, 1woman, solo, original adult character, short brown hair, green eyes, green cardigan, white shirt, seated, relaxed smile, holding a blue ceramic cup with both hands, window-side cafe table, potted plants, rainy street outside, waist-up view, eye-level angle, soft overcast window light, anime illustration, clean linework. An adult woman sits at a cafe table beside a rain-streaked window and holds a blue cup with both hands. Potted plants frame the window, with a wet street softly visible beyond it.
```

负向示例：`默认 CFG=1 时留空；有有效负面条件时使用上面的负向结构。`

Lowercase tags with spaces, then sentences for relationships. Keep scenes detailed enough to avoid a bare background. Do not apply non-Turbo step/CFG advice. Artist slot is absent unless requested, then use @artist.

依据：Family-supported short quality prefix, service slot composition. At CFG 1 the negative is a storage template only, not effective guidance.

[来源 1](https://huggingface.co/circlestone-labs/Anima)；[来源 2](https://civitai.com/models/2458426?modelVersionId=3263843)

## Anima Aesthetic v1.1

### 正向结构

```text
{count}, {character_and_series}, {appearance}, {clothes}, {pose_and_expression}, {action_and_spatial_relations}, {setting}, {composition}, {lighting}, {style}. {scene_sentences}
```

### 负向结构

```text
blurred contours, malformed hands, fused digits, inconsistent anatomy, unwanted lettering, watermark, signature, {scene_exclusions}
```

### 填写示例

```text
1woman, solo, original adult character, short brown hair, green eyes, green cardigan, white shirt, seated, relaxed smile, holding a blue ceramic cup with both hands, window-side cafe table, potted plants, rainy street outside, waist-up view, eye-level angle, soft overcast window light, anime illustration, clean linework. An adult woman sits at a cafe table beside a rain-streaked window and holds a blue cup with both hands. Potted plants frame the window, with a wet street softly visible beyond it.
```

负向示例：`blurred contours, malformed hands, fused digits, inconsistent anatomy, unwanted lettering, watermark, signature`

No score_* tags in either prompt and no automatic quality-word pile. Keep the example structure but fill the actual character, clothes, pose and scene. Do not apply Turbo defaults.

依据：Author-specific absence of score tags; the concrete scene order and negative are service defaults.

[来源 1](https://huggingface.co/circlestone-labs/Anima#aesthetic-version-prompting)；[来源 2](https://civitai.com/models/2458426?modelVersionId=3126581)

## Anima 2.9B Preview v1

### 正向结构

```text
best quality, detailed illustration, {count}, {character_and_series}, {appearance}, {clothes}, {pose_and_expression}, {action_and_spatial_relations}, {setting}, {composition}, {lighting}, {style} . {scene_sentences}
```

### 负向结构

```text
blurred contours, malformed hands, fused digits, inconsistent anatomy, unwanted lettering, watermark, signature, {scene_exclusions}
```

### 填写示例

```text
best quality, detailed illustration, 1woman, solo, original adult character, short brown hair, green eyes, green cardigan, white shirt, seated, relaxed smile, holding a blue ceramic cup with both hands, window-side cafe table, potted plants, rainy street outside, waist-up view, eye-level angle, soft overcast window light, anime illustration, clean linework. An adult woman sits at a cafe table beside a rain-streaked window and holds a blue cup with both hands. Potted plants frame the window, with a wet street softly visible beyond it.
```

负向示例：`blurred contours, malformed hands, fused digits, inconsistent anatomy, unwanted lettering, watermark, signature`

Keep the character name with the series when known. For multiple subjects assign each appearance and action explicitly. Score tags were absent from captions but permitted by the author; they are not a default here. Do not inherit MiaoMiao quality prefixes.

依据：Author supports detailed mixed captions and character-series pairing; short prefix and assembled slots are service defaults.

[来源 1](https://huggingface.co/Gazingstars123/Anima-2.9B)

## Krea2 Anime v3.0

### 正向结构

```text
An anime illustration of {subject}, with {appearance} and wearing {clothes}, {action}. The scene is set in {setting}; {spatial_relations}. {framing}, with {lighting}; {materials_and_color}. {requested_drawing_style}. {key_tags}
```

### 负向结构

```text
unwanted lettering, watermark, smeared outlines, {scene_exclusions}
```

### 填写示例

```text
An anime illustration of an adult woman with short brown hair and green eyes, wearing a green cardigan over a white shirt, holding a blue cup with both hands. She sits at a cafe table beside a rain-streaked window; potted plants stand behind the cup and the wet street is visible outside. An eye-level waist-up composition uses soft overcast window light, clear ceramic reflections and muted green accents. Clean ink outlines with gently layered colors. anime illustration, rainy cafe, blue ceramic cup
```

负向示例：`默认 CFG=1 时留空；有有效负面条件时使用上面的负向结构。`

Fill all useful scene sentences before adding 2–4 specific tags. No masterpiece/score tag prefix. Avoid generic claims like beautiful composition without concrete geometry. Use the drawing style requested by the user; do not invent an artist.

依据：Fine-tune author recommends rich prose plus strategic tags; this sentence skeleton and negative are service composition.

[来源 1](https://civitai.com/models/2908999?modelVersionId=3435333)；[来源 2](https://civitai.com/api/v1/models/2908999)

## Lustify Krea2 v10 Turbo file 3015314

### 正向结构

```text
{requested_medium} of {subject}, with {appearance}, wearing {clothes}, {action}. {framing_and_camera}; {spatial_relations_and_setting}. {lighting_direction_and_quality}, {materials_and_color}. {requested_style_or_finish}.
```

### 负向结构

```text
watermark, unwanted lettering, inconsistent anatomy, {scene_exclusions}
```

### 填写示例

```text
A natural-light portrait of an adult woman with short brown hair and green eyes, wearing a green cardigan and holding a blue ceramic cup. An eye-level waist-up view places her beside a cafe window, with plants behind her and a rainy street outside. Soft window light falls across her face and creates subdued reflections on the cup; the palette uses muted greens and warm wood. Gentle photographic contrast and realistic fabric texture.
```

负向示例：`默认 CFG=1 时留空；有有效负面条件时使用上面的负向结构。`

Describe visible subject, action, framing, background and light. Use photographic wording only for a photographic request; replace it with the requested illustration medium otherwise. Never infer sexual content from the model name. No SDXL/Pony score prefix. Do not confuse Turbo file 3015314 with Raw file 3015315.

依据：Fine-tune author has no published fixed prompt recipe. This is an explicitly service-designed Krea prose skeleton, not claimed author wording.

[来源 1](https://civitai.com/models/573152?modelVersionId=3112728)；[来源 2](https://huggingface.co/krea/Krea-2-Turbo)
