# Ötürmə sənədi — sıfırdan davam etmək üçün

> Yeni sessiya **yalnız bu faylı** oxumaqla işə başlaya bilər.
> Detallar: `IDEAS.md` (nə sınandı, nə öldü), `MODEL_V5.md` (hazırkı modelin
> tam dizaynı), `ARCHITECTURE.md` (v2 boru xətti), `WALKTHROUGH.md` (xronologiya).
>
> Vəziyyət: **2026-08-24**

---

## 1. Bir abzasda

GONG Hα tam-disk şəkillərində (2048×2048) günəş filamentlərinin instance
seqmentasiyası. Metrik **Panoptic Quality**. Deadline **15 Noyabr 2026**.
Ən yaxşı rəsmi bal **0.29**, lider **0.54**, dəstə 0.36–0.40.
İştirakçının hədəfi: **0.6+**.

---

## 2. HAZIRDA: v5 təlimdədir

**Fonda işləyir.** `train_v5.log`, checkpoint `checkpoints/spine_v5_last.pt`
(hər epoxada yazılır), ən yaxşısı `spine_v5_best.pt`.

```bash
# davam etdirmək (dayanıbsa)
./.venv/Scripts/python.exe -u src/spine_train3.py --epochs 36 --warmup 6 \
    --settle 6 --val-every 3 --val-limit 60 --tag spine_v5 \
    --resume checkpoints/spine_v5_last.pt --wait-for-gpu
```

**Vəziyyət (epoxa 10/36, epoxa ~60–80 dəq, qalan ~34 saat):**

| Yol | PQ (e9) | Qeyd |
|---|---|---|
| Piksel (U-Net) | **0.3191** | normal qalxır; v4 0.3495-ə çatmışdı |
| **Query head** | **0.0693** | TP 96, FP 668, FN 785 — **işləmir** |

**Qərar nöqtəsi: epoxa 15.** Query head hələ 0.15-dən aşağıdırsa, bu data
ölçüsündə (566 şəkil) işləməyəcək — onu atıb piksel yolu + yeni etiketlərlə
davam etmək lazımdır.

Self-training işləyir: 0.75-dən yuxarı filament sayı 2946 → 4336
(≈1400 promote). Düzgünlüyünü yalnız validasiya balı göstərəcək.

**Termal:** pik 79 °C (tavan 78). Qoruyucu dayandırır, 61–70 °C-yə qaytarır.
Davamlı deyil, amma sərhədə toxunur.

---

## 3. v5 nədir (tam dizayn `MODEL_V5.md`-də)

Altı dəyişiklik birdən — bilərək, çünki ayrı-ayrı sınamaq 6 gecə çəkərdi və
validasiya küyü (0.03) onları ayırd edə bilməzdi.

1. **Tam 2048 kadr**, kəsik yox (kəsik kənarları süni filament ucu öyrədirdi)
2. **18 giriş kanalı**: düzləşdirilmiş + **xam** şəkil, 5 fizika, stansiya (6),
   sinP/cosP, B0, ilin günü (2)
3. **Yeni etiketlər** — sayğac sistemi, aşağıda
4. **Self-training** — epoxa 7–30 arası model səs verir
5. **Xirallik başlığı** (`category_id`, indiyə qədər işlədilməyib)
6. **Query head** — obyekt səviyyəli, Hungarian matching

**Etiket sayğacı.** Bal = a/(a+b), hədd **0.75**. Başlanğıc çəki 5.

| Hal | Bal | Say |
|---|---|---|
| 1 of 3 | 0.40 | 386 |
| 1 of 2 | 0.50 | 395 |
| tək müəllif | 0.66 | 2187 |
| 2 of 3 | 0.80 | 268 |
| çarpaz-gün, **fərqli** müəllif | 0.90 | 1241 |
| hamı çəkib | 1.00 **donmuş** | 1395 |

Çarpaz-gün **eyni** müəllif: `a += 2.5`. Model səsi: hər epoxa `a += 1` və ya
`b += 1`, maksimum 30. İnsan etiketi döşəmədir.

**Səs meyarı:** filamentin içindəki cavab **ətraf fon**dan yüksəkdirmi.
Modelin öz hədəfi ilə müqayisə etmək olmaz — 0.40-a öyrədilən model 0.40 verir
və həmişə "yox" deyər, etiket sıfıra sürüşər.

---

## 4. Nəticələr lövhəsi

| # | Model | Validasiya | **Rəsmi** |
|---|---|---|---|
| 1 | Mask R-CNN (3 konfiqurasiya) | 0.000 | — |
| 2 | Spine U-Net, kalibrlənmiş | 0.3333 | 0.28 |
| 3 | eyni, mülayim kalibrləmə | 0.3098 | 0.24 |
| 4 | + fizika + konsensus + TTA (v2) | 0.3553 | **0.29** ← ən yaxşı |
| 5 | + zaman yayılması (v3) | 0.3593 | 0.28 ⚠️ sınıq |
| 6 | + Günəş ön-təlimi (v4) | 0.3790 / **0.3495** (60 dəst) | 0.27 |
| 7 | v5 | təlimdə | — |

**İstinad:** insan konsensusu **0.348** · boru xəttinin tavanı **0.734**.

⚠️ **Validasiya rəsmi baldan 0.05–0.09 yüksəkdir.** Və ölçü ilə dəyişir:
eyni model **25 dəstdə 0.3790, 60 dəstdə 0.3495**. Bütün qərarlar **60 dəstlə**.

---

## 5. Ən yaxşı modeli təkrar işlətmək (v2, rəsmi 0.29)

```bash
cd D:\Projects\Competitions\filament_detection
./.venv/Scripts/python.exe src/spine_predict.py --ckpt checkpoints/spine_v2_best.pt \
    --spine-thresh 0.95 --min-spine-px 600 --tta --out submission.csv
./.venv/Scripts/python.exe src/verify_submission.py submission.csv \
    --test-dir MAGFiLO_1.0_Kaggle_2026/test/test_images
./.venv/Scripts/kaggle.exe competitions submit -c filament-segmentation-2026 \
    -f submission.csv -m "<mesaj>"
```

⚠️ Submit **həmişə istifadəçinin qərarıdır**. Gündə 5 cəhd.

---

## 6. Qərarları idarə edən ölçmələr

| Nə | Rəqəm | Skript |
|---|---|---|
| İnsan konsensusu | PQ 0.348 | 156 cüt |
| Boru xəttinin tavanı | PQ 0.734 | `spine_infer.check_assembly()` |
| Filament forması | **8.2 × 125 px** | `check_spine.py` |
| Fotometrik düzəliş | kontrast 0.667σ → 1.331σ | `check_photometry.py` |
| Model recall | konsensus 71%, tək annotator **29%** | `check_annotator_effect.py` |
| Fırlanma proyeksiyası | **11–14 px = 0.9°** | `check_lookhere.py` |
| Çarpaz-gün bağlantı | 27.8% vs təsadüfi 8.9% = **3.1×** | `check_crossday_votes.py` |
| Zəif filamenti tapmaq | silmək **−0.019** | `check_single_vote_value.py` |
| Namizədlərin keyfiyyəti | boş həddə **39.5%** real | `check_instance_score.py` |
| Metrik doğruluğu | rəsmi kodla fərq 6e-10 | `check_official_scorer.py` |
| Tam kadr yaddaşı | 2048 batch 1 = 5798 MiB / 6144 | ölçüldü |

---

## 7. Rəsmi metrik

- PQ = Σ(TP IoU) / (TP + 0.5·FP + 0.5·FN), həd **IoU > 0.5**
- **`src/metrics.py` rəsmi kodla eynidir** (fərq 6e-10)
- GT `<annotator>-<şəkil>` formatındadır → **üç annotatorlu şəkildə bir proqnoz
  üç dəfə sayılır**
- Ölçüldü: tək müəllifin çəkdiyini proqnoz etmək **balı qaldırır**

---

## 8. Ölü istiqamətlər (təkrarlanmasın)

| İdeya | Səbəb |
|---|---|
| **Mask R-CNN / qutu-əsaslı** | 3 konfiqurasiya, hamısı PQ 0.000 |
| Zaman yayılması — formanı köçürmək | 87% qaranlıq olmayan yerə düşür, 0.29→0.28 |
| Qaranlıqlıq meyarı ilə "bura bax" | nəzarət 67% vs əsl 76% — **ayırd etmir** |
| Efemeridi düzəltmək | xəta artıq 0.9° — düzəldiləcək şey yoxdur |
| Fırlanma qanununu annotasiyalardan ölçmək | filamentləri tanıya bilmirik |
| Instance balı (əl əlamətləri) | +0.005, küy 0.03 — **sıfır** |
| Mülayim kalibrləmə | 0.28 → 0.24 |
| 1024 px-də təlim | filament eni 8→4 px |
| PINN / PDE loss | idarəedici PDE yoxdur |
| Ekvatordan ikiyə bölmək | vaxt qazandırmır, eyni piksel sayı |
| Tam transformer | 2048-də 16384 token, kvadratik; 566 şəkil azdır |

---

## 9. Növbəti addımlar

1. **Epoxa 15-də query head qərarı** (§2)
2. v5 bitəndə: kalibrləmə → proqnoz → yoxlama → submit (**icazə ilə**)
3. Query head işləməsə: piksel yolu + yeni etiketlərlə davam, sonra
   **YOLOv8-seg** müstəqil yol kimi (kod yoxdur, 2–3 gün iş)
4. Sınanmamış: dərin qatda attention; barb başlığı

---

## 10. Əməliyyat qeydləri

- **venv python məcburidir** — sistem python-unda torch CPU-onlydir
- GPU: **72% util tavanı, 78 °C-də dayandır**, bir iş bir anda
  (`python D:\Projects\Competitions\gpu_lock.py status`)
- Kaggle token `~/.kaggle/access_token` — **layihə faylına yazılmır**
- Commit-lər Claude-a istinad etmir
- Repo **15 Noyabra qədər private**, sonra public
- İstifadəçi: Günəş fizikası üzrə PhD. Fizikanı izah etmə, ölç və rəqəm göstər.
  **Qısa yaz.**
