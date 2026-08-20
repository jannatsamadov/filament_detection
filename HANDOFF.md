# Ötürmə sənədi — sıfırdan davam etmək üçün

> Yeni sessiya bu faylı oxumaqla işə başlaya bilər. Xronoloji hekayə
> `WALKTHROUGH.md`-də, texniki istinad `izah.md`-də, boru xəttinin izahı
> `ARCHITECTURE.md`-dədir.
>
> Vəziyyət: 2026-08-18

---

## 1. Bir abzasda

GONG Hα tam-disk şəkillərində (2048×2048) günəş filamentlərinin instance
seqmentasiyası. Metrik **Panoptic Quality**. Deadline **15 Noyabr 2026**.
Hazırkı ən yaxşı rəsmi bal **0.29**, leaderboard dəstəsi **0.36–0.40**, lider
**0.52**, 312 komanda. Qiymətləndirmə 70% kəmiyyət + 30% keyfiyyət.

İşləyən sistem: **U-Net (ResNet-34) + spine/radius/semantik başlıqlar**, tam 2048
ayırdetmədə, instance-lar spine komponentlərindən həndəsi qurulur.

---

## 2. Ən yaxşı modeli təkrar işlətmək

```bash
cd D:\Projects\Competitions\filament_detection
# həmişə venv python — sistem python-unda torch CPU-onlydir
./.venv/Scripts/python.exe src/spine_predict.py \
    --ckpt checkpoints/spine_v2_best.pt \
    --spine-thresh 0.95 --min-spine-px 600 --tta \
    --out submission.csv
./.venv/Scripts/python.exe src/verify_submission.py submission.csv \
    --test-dir MAGFiLO_1.0_Kaggle_2026/test/test_images
./.venv/Scripts/kaggle.exe competitions submit -c filament-segmentation-2026 \
    -f submission.csv -m "<mesaj>"
```

⚠️ Submit **həmişə istifadəçinin qərarıdır** — soruşmadan göndərilmir. Gündə 5 cəhd.

---

## 2b. HAZIRDA İŞLƏYƏN — v4 (2026-08-20)

**GPU-da təlim gedir.** Bitəndə: kalibrləmə → hesabat → proqnoz → yoxlama → submit.

```bash
# bitəndən sonra
./.venv/Scripts/python.exe src/spine_calibrate.py --ckpt checkpoints/spine_v4_best.pt     --tag spine_v4 --limit 25 --tta --spine-threshs 0.85 0.95     --min-spine-px 300 400 500 600 700
./.venv/Scripts/python.exe src/spine_predict.py --ckpt checkpoints/spine_v4_best.pt     --spine-thresh <tapilan> --min-spine-px <tapilan> --tta --out submission_v4.csv
./.venv/Scripts/python.exe src/verify_submission.py submission_v4.csv     --test-dir MAGFiLO_1.0_Kaggle_2026/test/test_images
# submit ISTIFADECININ qerarıdır
```

**v4 nədir:** v2 ilə **eyni**, yalnız başlanğıc çəkilər fərqlidir — ImageNet yerinə
**3772 etiketsiz GONG şəklində ön-təlim edilmiş** encoder.

**Ön-təlim** (`src/pretrain.py`, `checkpoints/pretrain_r34_encoder.pt`):
3772 şəkil (2010–2024, altı stansiyadan bərabər, `gong_unlabelled/`, hamısı eyni
fotometrik boru xəttindən keçib). Tapşırıq: şəklin 50%-ini 48-px bloklarla ört,
bərpa et. 20 epoxa, ~10 saat. Bərpa xətası 0.173 → 0.0349.

**v4 nəticələri (hər 4 epoxada, 4 hədd süpürməsi ilə):**

| Epoxa | 3 | 7 | 11 | 15 | 19 | 23 | 27 | **31** |
|---|---|---|---|---|---|---|---|---|
| PQ | 0.216 | 0.269 | 0.284 | 0.321 | 0.327 | 0.317 | 0.327 | **0.344** |
| FP | 286 | 238 | 200 | 170 | 138 | 190 | 146 | 150 |
| FN | 126 | 103 | 109 | 97 | 106 | 92 | 103 | **90** |

**Ön-təlim işləyir.** Eyni protokolla v3 ilə müqayisə:

| Epoxa | v3 (ImageNet) | **v4 (Günəş ön-təlimi)** |
|---|---|---|
| 7 | 0.241 | **0.269** |
| 15 | 0.276 | **0.321** |
| 19 | 0.293 | **0.327** |
| 31 | — (v3 max 0.324) | **0.344** |

v4 15-ci epoxada v3-ün 23-cü epoxasına çatır — **8 epoxa qabaqda**. Və **FN 90**,
v2-nin yekun iş nöqtəsində 119 idi: recall problemi düzəlməyə başlayıb.

Bu rəqəmlər hələ **TTA-sız və tam kalibrləməsizdir**; v2-də o iki addım
0.31 → 0.355 vermişdi.

⚠️ **v4-də zaman bərpası YOXDUR.** O, v3-də idi və uğursuz oldu (§6).

## 3. Nəticələr lövhəsi

| # | Model | Validasiya | **Rəsmi** |
|---|---|---|---|
| 1 | Mask R-CNN (3 konfiqurasiya) | 0.000 | — |
| 2 | Spine U-Net, kalibrlənmiş | 0.3333 | **0.28** |
| 3 | eyni, mülayim kalibrləmə | 0.3098 | **0.24** |
| 4 | **+ fizika kanalları + konsensus + TTA** | 0.3553 | **0.29** ← ən yaxşı |
| 5 | + zaman yayılması | 0.3593 | 0.28 ⚠️ sınıq, §6 |
| 6 | **+ Günəş ön-təlimi (v4)** | 0.344 (e31, TTA-sız) | təlimdə, §2b |

**İstinad nöqtələri:** insan konsensusu **0.348** · boru xəttinin tavanı **0.734**.

⚠️ **Validasiya rəsmi baldan sistematik 0.05–0.065 yüksəkdir** və **0.02-dən
kiçik fərqləri ayırd edə bilmir** — v3 validasiyada +0.004 yaxşı, rəsmi balda
−0.01 pis çıxdı. Bu ölçüdə "qazanclar" küydür.

---

## 4. Qərarları idarə edən ölçmələr

| Nə | Rəqəm | Mənbə |
|---|---|---|
| İnsan konsensusu (iki ekspert) | PQ 0.348, SQ 0.595, RQ 0.542 | `check` skripti, 156 cüt |
| Boru xəttinin tavanı | PQ 0.734, RQ 0.976 | `spine_infer.check_assembly()` |
| Filament forması | **8.2 × 125 px**, qutunun 24%-i | `check_spine.py` |
| Anchor örtməsi (IoU≥0.7) | standart **33%**, 3 ölçü ilə 70.5% | `check_anchors.py` |
| Spine ayrılması | 0.9% (<10 px), maskalarda 2.8% | `check_spine.py` |
| Fotometrik düzəliş | kontrast 0.667σ → **1.331σ** | `check_photometry.py` |
| Fibril kanalı | maska içi 15.6°, halqa 23.1° (təsadüfi 45°) | `check_physics_priors.py` |
| Oriyentasiya ↔ enlik | ρ = −0.152, **p = 8.6e-44** | `check_physics_priors.py` |
| Barblar | 81.5%-də var, median 18% sahə, ölçü ilə ρ=+0.409 | `check_barbs.py` |
| Etiket keyfiyyəti | annotasiyaların **0.02%**-i qaranlıq deyil | `check_label_quality.py` |
| Konsensus | 3 annotatorlu şəkillərdə piksellərin 30–50%-i **tək səs** | `precompute_consensus.py` |
| Model recall | konsensus filamentlərinin 71%-i, tək annotatorunkunun **29%**-i | `check_annotator_effect.py` |
| Fırlanma | 13.73°/gün ekvator, 10.93 @60°; **1° = 15.8 px** | `solar_ephemeris.py` |
| Stansiya parallaksı | **8.3 px** (0.92%) — əhəmiyyətsiz | `solar_ephemeris.py` |
| Metrik doğruluğu | rəsmi kodla fərq **6×10⁻¹⁰** | `check_official_scorer.py` |

---

## 5. Rəsmi metrik haqqında bilinənlər

Notebook `reference/self-evaluation-notebook.ipynb`-dədir (gitignore-da).

- PQ = Σ(TP IoU) / (TP + 0.5·FP + 0.5·FN), həd **IoU > 0.5**
- **Bizim `src/metrics.py` ona bərabərdir** (fərq 6e-10)
- GT **`<annotator>-<şəkil>`** formatındadır, proqnoz yalnız şəkil hissəsi ilə
  uyğunlaşdırılır → **üç annotatorlu şəkildə bir proqnoz üç dəfə sayılır**
- Deməli yalnız bir ekspertin gördüyü obyekt **1 TP + 2 FP** verir:
  metrik **birləşməni cəzalandırır, konsensusu mükafatlandırır**

---

## 6. ⚠️ Sınıq: zaman yayılması (v3)

İdeya doğrudur, **implementasiya sınıqdır** — nəticə çıxarılmır.

- Bərpa olunan filamentlərin **54.3%-i qaranlıq material üzərində deyil**
  (orijinal annotasiyalarda 0.0%)
- Kök səbəb: heliqrafik çevrilmənin **vaxt hissəsi**. Eyni vaxtlı kadrlar
  **0.0°** verir (fəza hissəsi düzgün), gün miqyaslı cütlərdə qalıq sürüşmə
  **median 3.5°**, bəzən 14.5°
- **3.5° = 55 piksel**, filament eni isə **8 piksel** → örtüşmə qalmır.
  Tələb olunan dəqiqlik **~0.5°**, yəni 7 dəfə yaxşı
- `L0` düzgün gedir (13.195°/gün), səhv onun tətbiqində və ya differensial
  fırlanma düzəlişindədir

**Metodiki dərs:** proyeksiya **eyni vaxta** round-trip ilə yoxlanmışdı (0.00 px).
Bu, çevrilmənin **tərsinə çevrilə bildiyini** yoxlayır, **düzgün olduğunu yox** —
səhv işarə konvensiyası özünü kompensasiya edir. Düzəltmək üçün `sunpy` kimi
**müstəqil** istinada qarşı yoxlanmalıdır.

Vizual sübut: `figures/13_rotation_derotated.gif` — filamentlər hələ sürüşür.

---

## 7. Ölü istiqamətlər (təkrarlanmasın)

| İdeya | Səbəb |
|---|---|
| **Mask R-CNN / qutu-əsaslı** | qutu-IoU 8×125 px əyri lentə uyğun gəlmir; 3 konfiqurasiya, hamısı PQ 0.000 |
| Kiçik anchor-lara keçmək | problem ölçü yox, **aralıq seyrəkliyidir** (33%→29%) |
| 1024 px-də təlim | filament eni 8→4 px, barblar itir |
| PINN / PDE loss | idarəedici PDE yoxdur |
| Həd+sentroid ilə disk tapmaq | halqanı Günəş sayır, radius 13% böyük |
| Radial profili median ilə | median filamentlərin özü tərəfindən dartılır |
| Mülayim kalibrləmə | 0.28 → 0.24 |
| Kalibrləmə optimumunu süpürmə kənarında seçmək | diapazon genişlənəndə optimum yerini dəyişir |
| Train annotasiyalarını test proqnozuna daşımaq | **qayda pozuntusu** |
| Şəkil müstəvisində zaman uyğunlaşdırması | 1 gündə 217 px = 1.7 filament uzunluğu |
| Fırlanma sürətini datadan ölçmək | iki üsul, ikisi də uğursuz; onsuz da təsiri 3–18 px |
| 0.02-dən kiçik val qazancını təqib etmək | test setinə keçmir |
| YOLO (sınanmayıb, amma) | maska başlığı 1/4 ayırdetmədə — 8 px obyekt 2 px olur; üstəlik **AGPL-3.0**, repo public olmalıdır |
| Nəzəri/sxematik filament şəkilləri əlavə etmək | domain fərqi çox böyük, GONG statistikası ilə əlaqəsizdir |
| GPU qoruyucusunda hər prosesi rəqib saymaq | NVIDIA App gecəlik run-ı bloklamışdı |

---

## 8. Növbəti addımlar, prioritetlə

**1. Recall — əsas darboğaz.** İş nöqtəsində TP 117, **FP 69, FN 119**. FP problemi
həll olunub (444→69), amma qiyməti recall oldu: **filamentlərin yarısını itiririk**.
Qazanc FP-ni geri qaytarmadan recall qaldırmaqdan gəlməlidir.

**2. ✅ EDİLDİ — etiketsiz GONG arxivində ön-təlim.** 3772 şəkil endirildi və
model onlarda ön-təlim edildi; nəticə §2b-dədir və müsbətdir. Uzatmaq olar
(20 epoxadan çox), yaxud **yalnız encoder** yükləməyi sınamaq olar — hazırda
decoder də yüklənir, o isə bərpa tapşırığına köklənib və v4 ilk epoxada v2-dən
pis başladı (loss 3.57 vs 2.76).

**3. Zaman yayılmasını düzəltmək — YENİ dizayn (iştirakçının təklifi).**
Köhnə üsul dünənki filamentin **surətini** çəkirdi; düzgün üsul zamanı yalnız
**"bura bax"** siqnalı kimi işlətməkdir, etiketi isə **bugünkü şəkildən** almaq.

⚠️ **Təlimdən ƏVVƏL yoxlama məcburidir** (bu dəfə edilməyib, ona görə 54% zibil
etiket fərq edilmədən təlimə getdi): yeni annotasiya faylı yaradılsın və
**neçə dəqiq filamentin şübhəli/qeydsiz filamentin üzərinə düşdüyü** sayılsın.
Rəqəm məntiqsizdirsə, təlim başlamamalıdır.

**4. Ansambl / daha uzun təlim.** Ən proqnozlaşdırıla bilən, amma ən az fərqləndirici.

**Hesabat və kod keyfiyyəti balın 30%-idir** — 15 Noyabr üçün 4 səhifəlik texniki
hesabat və pipeline notebook-u lazımdır.

---

## 9. Əməliyyat qeydləri

**GPU (kritik).** RTX 2060 6 GB, **fan işləmir**. Utilizasiya ≤72%, >78 °C pauza.
Ortaq `gpu_guard.Governor` və `gpu_lock.GpuLock` işlədilir (`D:\Projects\Competitions\`).
Güc limiti (`nvidia-smi -pl`) bu kartda **dəstəklənmir** — firmware kilidləyib.
Ölçülmüş profil: boşda 58 °C, 25% yükdə 71–74 °C, 55% yükdə 76–78 °C.
Epoxa ~9–11 dəq, 40 epoxa ~5–6 saat.

**Kaggle.** Token `~/.kaggle/access_token` (bütün layihələr üçün). Gündə 5 submission.
`./.venv/Scripts/kaggle.exe`.

**Repo.** `github.com/jannatsamadov/filament_detection`, **private**, 15 Noyabrda
public olmalıdır. Commit-lərdə **Claude atribusiyası yoxdur**.
`izah.md`, `WALKTHROUGH.md`, `HANDOFF.md`, `reference/`, data və checkpoint-lər
gitignore-dadır.

**Venv.** `./.venv/Scripts/python.exe` — sistem python-unda torch **CPU-onlydir**.

**Tələlər.** Windows konsolu cp1252 — skriptlərdə unicode print etmə.
`TaskStop` python prosesini öldürmür. Fon prosesləri shell `&` ilə başlatma.

---

## 10. Açıq suallar

| # | Sual | Niyə vacibdir |
|---|---|---|
| 1 | Test setində şəkil başına neçə annotator var? | metrik onları ayrıca sayır (§5), bu, FP büdcəsini müəyyən edir |
| 2 | IEEE BigData proceedings-də məqalə imkanı varmı? | yarışın dəyərini müəyyən edir |
| 3 | Hesablama səmərəliliyi necə ölçülür (70%-in hissəsi)? | model ölçüsü seçimi |
| 4 | Dəstə (0.36–0.40) hansı yanaşmadan istifadə edir? | bizim 0.29-la fərqin mənbəyi |

---

## 11. Fayl xəritəsi

**Sənədlər:** `HANDOFF.md` (bu) · `WALKTHROUGH.md` (xronologiya) ·
`izah.md` (texniki istinad) · `ARCHITECTURE.md` (boru xətti) · `README.md` (repo)

**Boru xətti:** `geometry` → `photometry` → `preprocess` → `physics_channels` →
`precompute_consensus` → `spine_data` → `spine_model` → `spine_train2` →
`spine_infer` → `spine_calibrate` → `spine_predict` → `verify_submission`

**Ölçmələr:** `check_*.py` — hər dizayn qərarının arxasındakı rəqəm

**Zaman bloku (sınıq):** `solar_ephemeris` · `filament_tracks` ·
`propagate_labels` · `measure_rotation{,2}` · `make_gif`
