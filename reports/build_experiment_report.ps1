$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$root = 'D:\liulanqi\code\ADAR'
$outDir = Join-Path $root 'reports'
$docxPath = Join-Path $outDir 'VisA-DR异常QA与定位实验整理报告.docx'
$pdfPath = Join-Path $outDir 'VisA-DR异常QA与定位实验整理报告.pdf'
$heatmapPath = Join-Path $root 'outputs\qwen3_visa_localization_20260921\candle_007.png'
$boxesPath = Join-Path $root 'outputs\qwen3_visa_roi_20260922\candle_007.png'
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Add()
    $doc.PageSetup.PaperSize = 2 # Letter
    $doc.PageSetup.TopMargin = 46
    $doc.PageSetup.BottomMargin = 46
    $doc.PageSetup.LeftMargin = 48
    $doc.PageSetup.RightMargin = 48
    $sel = $word.Selection

    function Set-BodyFont($size = 10, $color = 0x262626, $bold = $false) {
        $script:sel.Font.Name = 'Aptos'
        $script:sel.Font.NameFarEast = 'Microsoft YaHei'
        $script:sel.Font.Size = $size
        $script:sel.Font.Color = $color
        $script:sel.Font.Bold = [int]$bold
        $script:sel.ParagraphFormat.SpaceAfter = 5
        $script:sel.ParagraphFormat.LineSpacingRule = 0
    }
    function Add-Paragraph([string]$text, [string]$style = 'Normal', [int]$size = 10, [int]$color = 0x262626, [bool]$bold = $false) {
        Set-BodyFont $size $color $bold
        $script:sel.TypeText($text)
        $script:sel.TypeParagraph()
        $script:sel.ParagraphFormat.LeftIndent = 0
        $script:sel.ParagraphFormat.FirstLineIndent = 0
    }
    function Add-Heading([string]$text, [int]$level = 1) {
        Set-BodyFont $(if ($level -eq 1) { 16 } else { 12 }) 0x5D3617 $true
        $script:sel.ParagraphFormat.OutlineLevel = if ($level -eq 1) { 1 } else { 2 }
        $script:sel.ParagraphFormat.KeepWithNext = -1
        $script:sel.TypeText($text)
        $script:sel.TypeParagraph()
        $script:sel.ParagraphFormat.KeepWithNext = 0
    }
    function Add-Label([string]$text) {
        Set-BodyFont 10 0x5D3617 $true
        $script:sel.TypeText($text)
        $script:sel.TypeParagraph()
    }
    function Add-Sample([string]$label, [string]$text) {
        Add-Paragraph $label 'Normal' 9 0x5D3617 $true
        Set-BodyFont 9 0x262626 $false
        $script:sel.ParagraphFormat.LeftIndent = 18
        $script:sel.ParagraphFormat.RightIndent = 10
        $script:sel.ParagraphFormat.SpaceAfter = 7
        $script:sel.TypeText(('"' + $text + '"'))
        $script:sel.TypeParagraph()
        $script:sel.ParagraphFormat.LeftIndent = 0
        $script:sel.ParagraphFormat.RightIndent = 0
    }
    function Add-Table([string[]]$headers, [object[]]$rows) {
        $range = $script:sel.Range
        $table = $script:doc.Tables.Add($range, ($rows.Count + 1), $headers.Count)
        $table.Borders.Enable = 1
        $table.AllowAutoFit = $true
        $table.AutoFitBehavior(2) | Out-Null
        $table.Range.Font.Name = 'Aptos'
        $table.Range.Font.NameFarEast = 'Microsoft YaHei'
        $table.Range.Font.Size = 8.5
        $table.Range.Font.Color = 0x262626
        $table.Range.ParagraphFormat.SpaceAfter = 2
        $table.Range.Cells.VerticalAlignment = 1
        $table.Rows.AllowBreakAcrossPages = 0
        $table.Rows.Item(1).HeadingFormat = -1
        for ($c = 0; $c -lt $headers.Count; $c++) {
            $cell = $table.Cell(1, $c + 1)
            $cell.Range.Text = [string]$headers[$c]
            $cell.Shading.BackgroundPatternColor = 0x5D3617
            $cell.Range.Font.Color = 0xFFFFFF
            $cell.Range.Font.Bold = $true
            $cell.Range.ParagraphFormat.Alignment = 1
        }
        for ($r = 0; $r -lt $rows.Count; $r++) {
            for ($c = 0; $c -lt $headers.Count; $c++) {
                $cell = $table.Cell($r + 2, $c + 1)
                $cell.Range.Text = [string]$rows[$r][$c]
                if (($r % 2) -eq 1) { $cell.Shading.BackgroundPatternColor = 0xF2F5F9 }
                $cell.Range.ParagraphFormat.Alignment = if ($c -eq 0) { 0 } else { 1 }
            }
        }
        $script:sel.SetRange($table.Range.End, $table.Range.End)
        $script:sel.TypeParagraph()
    }
    function Add-PageBreak {
        $script:sel.InsertBreak(7) | Out-Null # wdPageBreak
    }
    function Add-Figure([string]$path, [string]$caption, [double]$widthInches = 6.25) {
        if (-not (Test-Path $path)) { Add-Paragraph "[图像文件缺失：$path]"; return }
        $picture = $script:sel.InlineShapes.AddPicture($path, $false, $true)
        $picture.LockAspectRatio = -1
        $picture.Width = $widthInches * 72
        $script:sel.TypeParagraph()
        Add-Paragraph $caption 'Caption' 8.5 0x666666 $false
    }

    # Cover and scope
    Set-BodyFont 22 0x5D3617 $true
    $sel.ParagraphFormat.Alignment = 1
    $sel.TypeText('VisA-D&R 异常 QA 与定位实验整理报告')
    $sel.TypeParagraph()
    Set-BodyFont 11 0x666666 $false
    $sel.ParagraphFormat.Alignment = 1
    $sel.TypeText('从文本相似度差异分析到区域提示验证｜实验记录核对版')
    $sel.TypeParagraph()
    $sel.TypeParagraph()
    $sel.ParagraphFormat.Alignment = 0
    Add-Paragraph '整理截止：2026-10-13    数据源：本地导出的逐题结果、服务器运行汇总与实验配置' 'Normal' 9 0x666666 $false

    Add-Heading '阅读说明与关键结论' 1
    Add-Paragraph '本报告按"实验标题—目标—具体结果—真实样例输出—结论与下一步优化"整理。纳入的数值均来自可核对的 CSV、JSON/JSONL、运行配置或服务器状态；没有完成的实验会单列为未完成，不以中间分数冒充最终成绩。'
    Add-Paragraph '主要范围：VisA-D&R 的 1,000 张异常图、4,029 个问答；Qwen2-VL-2B/7B、Qwen3-VL-4B 原生与 DeCo；主方法固定 N/A 标签前缀；视觉特征异常热图与候选框；ROI 图像 QA、历史上下文、oracle 与 text-only 对照；HDEC 区域扰动；MMAD/JUDO 风格评测进度。'
    Add-Table @('结论','证据') @(
        @('N/A 前缀没有改善文本指标','原生 Qwen3 与 DeCo 的 Low ROUGE-L、Low SBERT、Complex SBERT 均低于无前缀基线；主方法固定标签在该集合约 81.4% 正确。'),
        @('DeCo 对文本指标没有稳定全面增益','同一简洁提示下 Complex SBERT 上升约 0.0098，但 Low ROUGE-L、Low SBERT 下降；尚无显著性证据支持整体改善。'),
        @('定位图可给出区域线索，但不等于可靠框选','4-shot fusion_mean 像素 AUROC 0.884；top-1 peak-hit 仅 0.212。大框 top-k 覆盖提高，但图像面积开销也明显上升。'),
        @('ROI crop 的 1,000 图全量增益不稳健','fusion 对原图的 Complex SBERT +0.0044，95% 类别 bootstrap 区间跨 0；其他 3 个指标不升。'),
        @('当前 MMAD/JUDO 风格实验尚未完成','服务器仍在跑 Qwen3 原生端；最新指标快照 N=6,012/39,670，临时均值 77.11，不能作全 benchmark 成绩。')
    )
    Add-Heading '指标与协议边界' 2
    Add-Paragraph 'VisA-D&R 文本结果把 Q0–Q1 归为 Low-level，Q2 及之后归为 Complex。Low ROUGE-L 是预测/参考文本的 ROUGE-L F1 均值；Low SBERT 与 Complex SBERT 是 sentence-transformers/all-MiniLM-L6-v2 的余弦相似度均值；Complex ROUGE-L auxiliary 只作补充。它们衡量语言相似，不直接等于异常检测正确率、定位质量或事实正确率。'
    Add-Paragraph '协议存在代际差异：Qwen3 的"paper-ready"全量结果使用不超过 60 词的简洁英文指令、max_new_tokens=256、重复惩罚 1.2、no-repeat 3-gram；Qwen2 对照生成上限为 4096，且没有同一条 60 词指令。因此跨模型差异不能完全归因于模型架构。早期 Qwen3 长输出还出现截断，必须和后来的无截断版本分开看。'

    # Experiment 1
    Add-Heading '实验 1｜VisA-D&R 四模型文本问答基线与截断审计' 1
    Add-Label '目标'
    Add-Paragraph '量化 Qwen2/Qwen3 原生与 DeCo 的逐题文本相似度，查看长回答、否定语义与词面匹配如何共同影响指标，并保留真实输出供人工判断。'
    Add-Label '具体结果'
    Add-Paragraph '全量导出含每个模型 1,000 张图、4,029 个问题（2,000 个 Low-level pair、2,029 个 Complex pair）。下表采用已导出的逐题分数；Qwen3 两行来自通过质检的简洁版全量运行。'
    Add-Table @('模型','Low ROUGE-L','Low SBERT','Complex SBERT','截断记录*') @(
        @('Qwen2-VL-2B','0.2145','0.5962','0.5520','5 / 4,029'),
        @('Qwen2-VL-7B','0.2514','0.6441','0.5424','0 / 4,029'),
        @('Qwen3-VL-4B 原生','0.1391','0.6015','0.5594','clean run: 0'),
        @('Qwen3-VL-4B + DeCo','0.1346','0.5905','0.5692','clean run: 0')
    )
    Add-Paragraph '*Qwen2 截断计数来自逐题导出；Qwen3 简洁版的零截断与早期长输出轮次不是同一组生成。早期 Qwen3 长输出：原生 174/4,029（4.319%，109 张图受影响），DeCo 197/4,029（4.890%，119 张图受影响）；Qwen2-2B 为 5/4,029，Qwen2-7B 为 0。早期 Qwen3 平均输出约 707/741 token，触及 4,096 token 上限。不要把早期指标与后期 clean-run 指标拼接。' 'Normal' 8.5 0x555555 $false
    Add-Heading '真实样例输出｜candle/000.JPG，Q0' 2
    Add-Paragraph '标准答案：Yes, there is an anomaly in the upper left candle, where an irregular indentation is present on the wax surface.' 'Normal' 9 0x5D3617 $true
    Add-Sample 'Qwen2-VL-2B（ROUGE-L 0.2222；SBERT 0.5800）' 'There are no visible anomalies in the image of the candles.'
    Add-Sample 'Qwen2-VL-7B（ROUGE-L 0.1622；SBERT 0.6602）' 'No, there are no visible anomalies in the image of the candles. The candles appear to be evenly lit and there are no noticeable defects or irregularities.'
    Add-Sample 'Qwen3-VL-4B 原生（ROUGE-L 0.0714；SBERT 0.6489）' 'No obvious anomalies. All four candles appear uniform, with consistent shape and central wick placement. Lighting casts similar shadows, suggesting even illumination. No cracks, stains, or irregularities are discernible on their surfaces against the dark background. The arrangement is symmetrical and orderly.'
    Add-Sample 'Qwen3-VL-4B + DeCo（ROUGE-L 0.0727；SBERT 0.5912）' 'No significant anomalies detected. All four candles appear identical—round, white, with central wicks—and are evenly spaced on a dark surface under consistent lighting conditions. Shadows and slight variations due to lighting are normal for such objects. No defects or irregularities observed.'
    Add-Paragraph '同一图像 Q1 的标准答案认为凹陷是异常；DeCo 回答 "The indentation isn''t necessarily abnormal—it''s likely intentional design for holding the wick securely…"，但 SBERT 仍为 0.8263。它说明 SBERT 可能因主题词和语义场相近而给出高分，即使结论方向相反；ROUGE-L 则会惩罚改写、补充和长度差异。'
    Add-Label '结论与下一步优化'
    Add-Paragraph '数值显示 Qwen2 的词面重合高于 Qwen3，但并不表示 Qwen2 更理解缺陷；例子中多模型都否认标准答案指出的异常。下一步先用同一 tokenizer 级长度约束、同一 prompt、同一解码策略重跑受控模型对照，再把"语义相似"与"异常判断正确性/事实一致性"分开报告。'

    # Experiment 2
    Add-Heading '实验 2｜在 Q0 前加入主方法 N/A 判断先验' 1
    Add-Label '目标'
    Add-Paragraph '检验在第一问前给模型一个由主方法 HDEC + Youden 产生的 "This picture is normal/abnormal." 标签，是否能让后续回答更贴近 VisA-D&R 参考答案。回答仍由 Qwen3 生成；该标签不参与重新评分类指标。'
    Add-Label '具体结果'
    Add-Paragraph '先验试验覆盖 1,000 张图、4,029 个问题。主方法标签与数据集真值的准确率为 814/1,000（81.4%），约 18.6% 标签错误；该固定标签并非完美提示。下表的无前缀基线使用相同的 Qwen3 clean-run 汇总作为对照，但先验分支自身输出上限/QC 与基线略有差异，故 delta 是观测差而非严格隔离因果效应。'
    Add-Table @('模型/条件','Low ROUGE-L','Low SBERT','Complex SBERT','相对无前缀变化') @(
        @('Qwen3 原生｜无前缀','0.139060','0.601543','0.559374','基线'),
        @('Qwen3 原生｜加主方法标签','0.121913','0.597893','0.542127','−0.017147 / −0.003650 / −0.017247'),
        @('Qwen3 + DeCo｜无前缀','0.134577','0.590544','0.569162','基线'),
        @('Qwen3 + DeCo｜加主方法标签','0.116197','0.585598','0.541791','−0.018380 / −0.004946 / −0.027371')
    )
    Add-Paragraph '原生先验运行没有截断；DeCo 先验运行有 3 条触及 token cap、2 条退化输出，QC 未通过。先验运行配置中主方法标签来自既有 detection_cases.jsonl；Qwen3 原生与 DeCo 的生成上限也不完全一致，不能把所有差值都归因于标签本身。' 'Normal' 8.5 0x555555 $false
    Add-Heading '真实样例输出｜candle/000.JPG' 2
    Add-Paragraph '真值：异常。主方法得分 −9.270061，按类别 Youden 阈值 −7.668632 判为 NORMAL，因此输入提示为 "This picture is normal."' 'Normal' 9 0x5D3617 $true
    Add-Sample 'Qwen3 原生 Q0' 'No visible anomalies. The four white tea light candles are evenly arranged on a dark surface with consistent shadows and minor imperfections typical for such objects. No unusual shapes, colors, or distortions present that indicate defects or abnormalities.'
    Add-Sample 'Qwen3 原生 Q1' 'It''s not truly an anomaly—such indentations are standard during manufacturing to hold wicks. However, if unusually deep or misshapen compared to others, it might be seen as irregular due to inconsistent production quality, but isn''t inherently defective unless specified otherwise.'
    Add-Paragraph '该例展示错误先验会把后续文字带向"正常"解释；另一个主方法判为异常的 candle 样例中，Qwen3 与 DeCo 都能接着描述中央缺蜡孔，但全量指标没有改善。'
    Add-Label '结论与下一步优化'
    Add-Paragraph '在当前实现下，先验没有改善三项文本相似度，且会传播主方法自身约 18.6% 的标签错误。若继续研究，须把标签作为独立条件变量做成匹配四格设计（正确/错误标签 × 原生/DeCo），并锁定相同 generation cap；不能把简单前缀作为指标提升方案。'

    # Experiment 3
    Add-Heading '实验 3｜视觉特征异常响应图与候选区域框' 1
    Add-Label '目标'
    Add-Paragraph '把 Qwen3-VL 的视觉表征转成 patch 级异常热图，评估不同层表征对像素级缺陷区域的响应；再把高分区转成候选框，判断其能否支撑后续 ROI QA。此实验属于视觉定位诊断，不是文本相似度实验。'
    Add-Label '具体结果'
    Add-Paragraph '在 10 个 VisA 类别上，使用每类 4 张正常支持图建立参考库；测试每类最多 100 张正常与 100 张异常图（最多 2,000 张），GT mask 只用于评估。256×256 网格、不做 per-image score normalization。'
    Add-Table @('表征','Pixel AUROC','Peak-hit rate') @(
        @('deepstack_5','0.9135','0.173'),
        @('deepstack_11','0.7997','0.048'),
        @('deepstack_17','0.7885','0.094'),
        @('final','0.8196','0.042'),
        @('fusion_mean','0.8842','0.212'),
        @('fusion_max','0.8028','0.047')
    )
    Add-Paragraph 'Pixel AUROC 较高表示异常像素总体排序有区分度；peak-hit 只有 4.2%–21.2%，说明最高响应点并不稳定落在 GT 缺陷内。ROI proposal 方面，fusion_mean 的 top-3 64×64 框覆盖平均为 78.15%，至少 90% 缺陷覆盖的图像占 70.6%，但框面积约占整图 17.46%；扩大到 top-5 96×96 后，90% 缺陷覆盖率升至 91.5%，但框面积升至 55.2%。'
    Add-Figure $heatmapPath '图 1. candle 样例的输入、GT mask 与 final/fusion_mean 热图。热图响应较弥散，局部缺陷有响应，但背景/主体区域也明显激活。' 6.25
    Add-Figure $boxesPath '图 2. 候选框和 GT mask 对照。此例中不同层的 top-k 框未完全覆盖小缺陷，说明候选框仍需校准。' 6.25
    Add-Label '结论与下一步优化'
    Add-Paragraph '视觉响应有可用信号，但"像素排序好"不等于"框完整且准确"。当前结果也不是与公开 SOTA 同协议比较，不能宣称达到定位 SOTA。下一步优先在独立测试集报告 pixel-AUROC、AUPRO、区域召回和框面积/覆盖 trade-off，先稳定定位，再将预测 ROI 作为 QA 输入。'

    # Experiment 4
    Add-Heading '实验 4｜原图、随机裁剪与特征引导 ROI 裁剪的 QA' 1
    Add-Label '目标'
    Add-Paragraph '检验把视觉特征挑出的区域裁剪与原图一起输入，是否能提高 Qwen3 对参考文本的 Low/Complex 相似度。全量阶段 1,000 张图，三臂各生成 4,029 个回答，共 12,087 个回答，生成约 31,013 秒（约 8.6 小时），无重试失败。'
    Add-Label '具体结果'
    Add-Table @('条件','Low ROUGE-L','Low SBERT','Complex SBERT','Complex ROUGE-L aux') @(
        @('原图','0.139008','0.601286','0.563488','0.134042'),
        @('随机 3 crops + 原图','0.121925','0.595367','0.560908','0.131597'),
        @('fusion_mean 3 crops + 原图','0.136511','0.595530','0.567813','0.133033')
    )
    Add-Paragraph '按类别配对 bootstrap 比较 fusion 与原图：Low ROUGE-L Δ −0.00250，95% CI [−0.00627, 0.00126]；Low SBERT Δ −0.00576，CI [−0.01708, 0.00462]；Complex ROUGE-L aux Δ −0.00100，CI [−0.00404, 0.00231]；Complex SBERT Δ +0.00436，CI [−0.00229, 0.01131]。四个区间均跨 0。三臂同为 3 个 crop；但原图臂视觉 token 更少，因此 fusion 对 random 的控制更公平。'
    Add-Paragraph '先导 20 图试验中 fusion 相对原图的 Low ROUGE-L +0.00571、Low SBERT −0.00433、Complex SBERT +0.01976、Complex ROUGE-L aux +0.00701；该 pilot 与全量集合重叠，不能当作独立重复。'
    Add-Heading '真实样例输出｜candle/025.JPG，Q0' 2
    Add-Paragraph '参考答案：Yes, the lower right candle shows a damaged area characterized by an irregular, darker section.' 'Normal' 9 0x5D3617 $true
    Add-Sample '原图（ROUGE-L 0.1538；SBERT 0.6420）' 'Yes, one candle on the bottom right has a noticeable chip near its edge, while others appear intact and uniform. The lighting casts consistent shadows across all candles, indicating even illumination from above. No other irregularities are evident.'
    Add-Sample '随机裁剪（ROUGE-L 0.1481；SBERT 0.6272）' 'Yes, one candle on the bottom right has a noticeable chip near its edge and appears slightly misshapen compared to others. All candles have similar central wick holes but vary subtly in texture and lighting shadows cast against the dark background.'
    Add-Sample 'fusion_mean 裁剪（ROUGE-L 0.1538；SBERT 0.6703）' 'Yes, one candle shows a noticeable chip on its edge and another has minor surface imperfections. All candles appear otherwise uniform with standard wick placements against dark background lighting casting distinct shadows. No other significant anomalies detected.'
    Add-Label '结论与下一步优化'
    Add-Paragraph '裁剪方案相对随机裁剪有所恢复，但全量相对原图没有可靠增益。当前不建议把 ROI crop 宣称为文本指标改进方法；若继续，应固定视觉 token 预算、用独立类别/图像评估，并把定位误差与答案变化联合分析。'

    # Experiment 5
    Add-Heading '实验 5｜多轮问答历史上下文消融' 1
    Add-Label '目标'
    Add-Paragraph '区分连续问答历史中"保留之前的问题"和"保留模型之前的回答"对后续回答的影响，尤其是 Complex QA 中对异常原因、预防方式的回答。验证集 100 张图；两种历史条件和原图/随机 crop/fusion crop 均使用同一图像集合。'
    Add-Label '具体结果'
    Add-Table @('历史设置','视觉输入','Low ROUGE-L','Low SBERT','Complex SBERT') @(
        @('Independent questions','原图','0.140261','0.579401','0.546040'),
        @('Independent questions','随机 crop','0.135757','0.585738','0.538171'),
        @('Independent questions','fusion crop','0.139282','0.593807','0.552633'),
        @('Questions-only history','原图','0.139743','0.590310','0.589185'),
        @('Questions-only history','随机 crop','0.129651','0.586551','0.585644'),
        @('Questions-only history','fusion crop','0.141776','0.589415','0.587795'),
        @('同 100 图的 full-history 基线子集','原图','0.136996','0.598970','0.556187')
    )
    Add-Paragraph 'Independent questions 会移除此前 QA 问题及模型回答；Questions-only history 保留先前 benchmark 问题文本，不保留模型先前回答。它不仅改变"是否携带答案"，也改变上下文框架，因此不是对单一变量的完美隔离；每个条件仅 100 图，未给出置信区间。'
    Add-Heading '真实样例输出｜candle/024.JPG，Q2' 2
    Add-Paragraph '问题：What could have caused this anomaly with the candle?' 'Normal' 9 0x5D3617 $true
    Add-Sample 'Independent questions（没有之前问题/回答）' 'The candle likely had its wick improperly positioned or was handled roughly during manufacturing, causing a small indentation or misalignment on one side, affecting symmetry and potentially burning unevenly when lit.'
    Add-Sample 'Questions-only history（保留此前问题，不保留模型回答）' 'The candle''s anomaly likely stems from a manufacturing defect—perhaps an uneven wax layer or misaligned wick—which causes irregularities on its surface and may affect how it burns or holds shape during use.'
    Add-Label '结论与下一步优化'
    Add-Paragraph '历史提示能改变输出内容与复杂问题的相似度，但方向并不一致，且目前样本不足以断言哪种历史设计普遍更好。下一步固定图像、问句顺序、参考图与解码设置，做同图配对、多随机种子，并把历史文本长度作为控制变量。'

    # Experiment 6
    Add-Heading '实验 6｜GT-mask oracle crop 与 text-only 负控制' 1
    Add-Label '目标'
    Add-Paragraph '用 GT mask 选 crop 估计"如果定位近乎完美，文本指标是否可能提高"；再完全移除图像做 text-only 对照，衡量模型仅凭问题/上下文产生的语言先验。两项各 100 张图、400 个回答，不是全量 benchmark。'
    Add-Label '具体结果'
    Add-Table @('条件（100 图）','Low ROUGE-L','Low SBERT','Complex SBERT','说明') @(
        @('匹配子集原图','0.136996','0.598970','0.556187','同一 100 图的 full-history 原图基线'),
        @('GT-mask oracle crops','0.141130','0.604395','0.585019','均值 mask 覆盖 99.22%；非部署方案'),
        @('Text-only（无任何图像张量）','0.115584','0.553779','0.545043','语言/上下文负控制')
    )
    Add-Paragraph 'Oracle 相对同 100 图基线的 Complex SBERT +0.02883，95% CI [+0.01093, +0.04750]；Low ROUGE-L Δ +0.00413，CI [−0.00577, +0.01391]；Low SBERT Δ +0.00543，CI [−0.01211, +0.02251]。与 fusion crop 比较时，Complex SBERT +0.01640，CI [−0.00050, +0.03316]，区间跨 0。GT mask 是 oracle 选择信息，不能作为可部署方法分数。Text-only 相对匹配基线分别低 0.02141 / 0.04519 / 0.01114。'
    Add-Heading '真实样例输出｜candle/024.JPG，Q0' 2
    Add-Paragraph '标准答案：Yes, there is an anomaly on the candle at the lower left. It has a jagged tear with a metallic sheen along its right edge.' 'Normal' 9 0x5D3617 $true
    Add-Sample 'GT-mask oracle crop' 'Yes, one candle''s edge shows a small chip or imperfection where wax has broken away slightly, revealing an underlying layer. This minor flaw contrasts with the otherwise uniform appearance of all four candles on the dark surface.'
    Add-Sample 'Text-only，无图像' 'No obvious anomalies detected. Image appears clear and consistent with expected visual content. No distortions, missing elements, or irregularities stand out upon quick review. All features align as normal without signs of corruption or unexpected artifacts.'
    Add-Paragraph '同一 text-only 样例的 Q1 还凭空描述"candle''s flame flickers unnaturally"；真实图中是未点燃的蜡烛。该例是模型在没有视觉输入时使用语言先验的直接迹象。'
    Add-Label '结论与下一步优化'
    Add-Paragraph 'Oracle 结果支持"精准视觉区域可能有助于复杂语义回答"的假设，但无法证明当前自动定位已达到该效果；text-only 明显下降，也会发生不符合图像的幻觉。下一步应优先提高预测 ROI 的 mask/box 质量，再做部署可用的自动 crop 与 oracle 上界对照。'

    # Experiment 7
    Add-Heading '实验 7｜HDEC 原始分数的 ROI 删除/模糊扰动' 1
    Add-Label '目标'
    Add-Paragraph '检验视觉特征提出的 ROI 是否对主方法原始 HDEC 分数有比随机区域更大的影响。该实验改动局部图像后重算分数，不生成文本，也不直接计算 AUROC、ACC 或文本相似度。'
    Add-Label '具体结果'
    Add-Paragraph '扩展组为 200 张图（100 normal、100 anomaly），从 4-shot feature map 取 top-1 shallow/deepstack_5 或 fusion_mean 区域，并与随机区域对照；使用 blur 或 image-mean 填充。表中 score drop = 原图分数 − 扰动后分数；advantage = ROI mean drop − random mean drop。'
    Add-Table @('真值 / 填充','定位方法','ROI mean drop','随机 mean drop','配对优势','ROI > 随机') @(
        @('Anomaly / blur','fusion_mean','+1.0595','−0.8267','+1.8862','79 / 100'),
        @('Anomaly / image-mean','fusion_mean','+1.1413','−0.8082','+1.9495','76 / 100'),
        @('Normal / blur','fusion_mean','−1.5354','−2.5879','+1.0525','69 / 100'),
        @('Normal / image-mean','fusion_mean','−1.7235','−2.7616','+1.0381','70 / 100'),
        @('Anomaly / blur','deepstack_5','+0.3095','−0.8267','+1.1362','65 / 100')
    )
    Add-Paragraph '对异常图，fusion_mean 区域扰动通常比随机扰动更明显地降低原始异常分数；但填充方式、框面积和位置本身也会改变视觉输入，且正常样本上的变化不等于分类收益。这个结果只是局部敏感性诊断，不是"因果证明模型看到了缺陷"。'
    Add-Label '结论与下一步优化'
    Add-Paragraph '结果为视觉区域与 HDEC 分数的关联提供了定量证据，但不能替代独立像素定位评估或 QA 指标。下一步可加入多个填充策略、区域尺寸匹配、遮挡与反事实置换，并在固定 Youden 阈值下单独计算 ACC/Recall/Precision/F1 的变化。'

    # Experiment 8 and pending work
    Add-Heading '实验 8｜MMAD / JUDO 风格问答基准：当前运行状态' 1
    Add-Label '目标'
    Add-Paragraph '在完整 MMAD 多选 QA 测试集上按 1-shot 方式比较 Qwen3-VL-4B 原生与 Qwen3-VL-4B + DeCo，并使用 benchmark 的任务类别与总体 accuracy 口径；它与 VisA-D&R 自由文本 ROUGE/SBERT 不是同一 benchmark。'
    Add-Label '具体结果（截至 2026-10-13 状态检查）'
    Add-Table @('项目','状态/数值','解释') @(
        @('测试集预检','39,670 QA；8,366 张 query 图；6,114 张 normal template；0 张引用图片缺失','MMAD JSON 比论文报告的 39,672 题少 2 题，需在最终结果中披露。'),
        @('原生 Qwen3 进度','最新指标快照 6,012 / 39,670；Python 评测进程仍运行','状态查询时 JSONL 继续增长；DeCo 尚未形成可比较的完整汇总。'),
        @('临时汇总','77.11；parse-valid 97.67%；Overkill 2.02%；Miss 41.06%','最新临时 CSV 标记 MVTec-AD，分组仍需完成后复核；不能当作全 benchmark 均值。'),
        @('最终对比','未完成','目前没有可报告的 Qwen3 原生 vs DeCo 全量结果。')
    )
    Add-Paragraph '运行配置为 query image + 1 张同类 normal template，16×16 网格提示，Qwen3-VL-4B-Instruct，max_new_tokens=1024；DeCo 分支设 α=0.6、层 20:29。该配置是 JUDO-style benchmark protocol，不等于 JUDO 模型复现。'
    Add-Heading '论文结果参照（不是本实验结果）' 2
    Add-Table @('方法 / 论文','MMAD 平均准确率','说明') @(
        @('MMAD 论文 GPT-4o','74.9%','ICLR 2025 论文报告值'),
        @('JUDO Qwen2.5-VL-7B base','72.56%','JUDO ICLR 2026 Table 1；标准 1-shot'),
        @('JUDO','81.20%','训练后 7B 模型；不是纯推理期提示法')
    )
    Add-Paragraph 'JUDO 使用 Qwen2.5-VL-7B，并通过比较式分割 SFT、领域知识 SFT 与 GRPO 训练；我们的本次实验是未对 Qwen3 做这种训练，只改变解码过程（DeCo）并按 1-shot 输入 normal template。因此最终可以做 benchmark 上的分数参照，但不能称为"复现了 JUDO 方法"。'
    Add-Label '结论与下一步优化'
    Add-Paragraph '先等待原生和 DeCo 两臂都完成，再从原始 JSONL 重新按官方任务定义复算 parse-valid accuracy、严格把无效选项计错的 accuracy，并核对七任务宏平均；确认样本数和分组后才与 74.9/72.56/81.20 比较。当前 77.11 只作进度信息，不进入结论。'

    # Literature and unrun items
    Add-Heading '尚未完成或不能直接比较的项目' 1
    Add-Table @('项目','当前证据','报告处理') @(
        @('GPT-Score / GPT-4 judge','本地汇总中的 low_gpt_score、complex_gpt_score 为空；没有 API 评审记录','标记未完成；未用本地模型冒充 GPT-4 judge。'),
        @('Anomaly-OV 官方模型复跑','没有可用的官方权重/复现输出；已有用户提供论文表中的数值','仅列论文表格参考值，不称为本地复现。'),
        @('Anomaly-OV 论文表参考','0.5B：Low ROUGE-L .33 / Low SBERT .72 / Complex SBERT .83；7B：.34 / .73 / .84','不同论文生成和评估流程；不能与本实验直接作同条件因果比较。'),
        @('JUDO 方法复现','当前仅运行 Qwen3 1-shot benchmark 流程，JUDO 的三阶段训练未复现','用"benchmark 对照"称呼，不写"方法复现"。'),
        @('定位 SOTA 比较','当前 ROI 是本地 4-shot patch 特征实验，分辨率/指标口径不同','没有直接 SOTA 名次；后续统一 mask、AUPRO、pixel-AUROC、数据划分再比较。')
    )
    Add-Heading '综合结论与下一步优先级' 1
    Add-Paragraph '当前最可靠的结论不是"某个提示能把三个文本指标显著拉高"，而是：语言生成方式对 ROUGE-L/SBERT 数值影响大；固定 N/A 先验没有带来改善；DeCo 在 Qwen3 上只呈现指标间权衡；区域输入的实际收益依赖定位质量，oracle 上界高于当前自动 crop；text-only 会产生与图像不符的流畅解释。'
    Add-Table @('优先级','下一步','为什么') @(
        @('1','锁定统一文本协议并重跑可控基线','同一模型输入、问句、上下文、最大输出长度与解码参数，才能解释模型/DeCo差异。'),
        @('2','完成 MMAD 原生和 DeCo 全量评测与数据复核','先得到可复现的七任务 benchmark 主结果，再做论文对照。'),
        @('3','提升可部署定位而不是继续加裁剪提示','peak-hit 偏低、proposal 的精确性未充分验证；oracle 结果说明这是关键瓶颈。'),
        @('4','将语言相似度与答案正确性分表报告','补充判断正确率/证据一致性或经人工核验指标，不让 SBERT 高分掩盖否定答案。'),
        @('5','正式运行 GPT judge 或明确不做','有 API 与版本/提示记录后才填 GPT-Score；否则保留空项。')
    )
    Add-Heading '数据与出处' 1
    Add-Paragraph '本地来源（相对 ADAR 根目录）：qwen_visa_dr_1000_analysis/export/{model_summary.csv, all_models_qa_metrics.csv, category_examples.json}；visa_dr_reasoning_all_models_summary_20260907/{metrics_summary.csv, truncation_summary.csv}；qwen3_visa_roi_qa_full_20260922/{summary.json, paired_category_bootstrap.json}；qwen3_visa_localization_20260921/macro_localization.csv；qwen3_visa_roi_20260922/macro_roi.csv；qwen3_visa_roi_qa_oracle100_20260927/{summary.json, audit.json}；qwen3_visa_roi_qa_textonly100_20260929/summary.json；hdec_roi_deletion200_20260929/summary.json。' 'Normal' 8.5 0x262626 $false
    Add-Paragraph '服务器来源：visa_dr_qwen3_simple_prior_full_20260914_113649；qwen3_visa_roi_qa_independent_validation100_20260924；qwen3_visa_roi_qa_questions_only_validation100_20260924；mmad_judo_qwen3_pair_20261013。MMAD 状态是 2026-10-13 运行快照。' 'Normal' 8.5 0x262626 $false
    Add-Paragraph '论文参照：MMAD, "Multi-modal Anomaly Detection," ICLR 2025, https://openreview.net/pdf?id=JDiER86r8v；JUDO, "A Juxtaposed Domain-Oriented Multimodal Reasoner for Industrial Anomaly QA," ICLR 2026, https://proceedings.iclr.cc/paper_files/paper/2026/file/92a7a03e1c716970848a4a86cc8243ee-Paper-Conference.pdf。Anomaly-OV 数值来自用户提供的论文表图，未本地复跑。' 'Normal' 8.5 0x262626 $false

    # Footer with page number
    foreach ($section in $doc.Sections) {
        $footer = $section.Footers.Item(1).Range
        $footer.Text = 'ADAR 实验归档    •    2026-10-13    |    第 '
        $footer.ParagraphFormat.Alignment = 1
        $footer.Font.NameFarEast = 'Microsoft YaHei'
        $footer.Font.Name = 'Aptos'
        $footer.Font.Size = 8
        $footer.Font.Color = 0x777777
        $footer.Collapse(0)
        $footer.Fields.Add($footer, 33) | Out-Null # wdFieldPage
    }

    $doc.SaveAs2($docxPath, 16) | Out-Null # wdFormatDocumentDefault
    $doc.ExportAsFixedFormat($pdfPath, 17) | Out-Null # wdExportFormatPDF
    $doc.Close(0)
    try { $word.Quit() } catch { }
    $sel = $null
    $doc = $null
    $word = $null
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
    Write-Output "DOCX=$docxPath"
    Write-Output "PDF=$pdfPath"
} finally {
    if ($doc -ne $null) { try { $doc.Close(0) } catch {} }
    if ($word -ne $null) { try { $word.Quit() } catch {} }
}
