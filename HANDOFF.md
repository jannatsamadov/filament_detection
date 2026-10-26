# Ötürmə sənədi — sıfırdan davam etmək üçün

> Yeni sessiya **yalnız bu faylı** oxumaqla işə başlaya bilər.
> Detallar: `IDEAS.md` (nə sınandı, nə öldü), `MODEL_V5.md` (hazırkı modelin
> tam dizaynı), `ARCHITECTURE.md` (v2 boru xətti), `WALKTHROUGH.md` (xronologiya).
>
> Vəziyyət: **2026-08-24**

---

## 0. İndi gedən iş (2026-10-26)

**Təlim artıq Kaggle-da gedir, yerli GPU-da yox** (pulsuz 2×T4, həftədə 30 saat).
`kernels/aug_unet/run.py`: güclü augmentasiya (`full`) və yalnız flip (`flip`)
bir sessiyada, hər biri öz kartında, 60 epoxa (~4.7 saat). Dizayn və ölçmələr:
`izah.md` §15.

```bash
./.venv/Scripts/kaggle.exe kernels status jannatsamadov/filament-aug-unet
./.venv/Scripts/kaggle.exe kernels output jannatsamadov/filament-aug-unet -p kernels/aug_unet/out_kaggle
./.venv/Scripts/python.exe kernels/aug_unet/report.py      # -> figures/aug_unet_report.png
./.venv/Scripts/python.exe src/verify_submission.py kernels/aug_unet/out_kaggle/submission_full.csv \
    --test-dir MAGFiLO_1.0_Kaggle_2026/test/test_images
```

Leaderboard-un 0.55–0.62 topası açıq notebook-dakı hazır CSV-dən gəlir, model
deyil (`izah.md` §15). Tələbənin sadə modeli 0.38 alıb — bizim v2-dən (0.29) yüksək.

**Kernel zənciri** (hamısı `kernels/`-də; tək-fayllı kernel-ləri `python kernels/build.py` qurur):

| kernel | nə edir | mənbə |
|---|---|---|
| `filament-aug-unet` | run 1: `flip` vs `full`, 60 epoxa | yarış datası |
| `filament-aug-dump` | run 1 checkpoint-lərindən bütün val/test xəritələri (plain + D4 TTA) | `filament-aug-unet` |
| `filament-aug-post` | **yalnız CPU**: geniş grid, ansambl, öyrədilmiş keep-rule, hər variant üçün CSV | `filament-aug-dump` (+ sonra `filament-unet-r2`) |
| `filament-unet-r2` | run 2: `union` hədəf və ResNet-50, güclü augmentasiya üstündə; xəritələri özü yazır | yarış datası |

**Nəticə (2026-10-26 10:20, `izah.md` §15 sonu):** 4 model, ən yaxşı val PQ
0.4400 (ansambl r50+flip, D4, keep-rule), ən yaxşı tək model R50+D4 0.4335
(PQ60 0.4004). Augmentasiya, union hədəf, R50 — hamısı küy (<0.02) daxilində;
yalnız D4 TTA ardıcıl kömək edir. **Hələ submit edilməyib** — yoxlanmış CSV-lər
`kernels/aug_post/out_kaggle_r2/` (`submission_ens-r50-flip_d4_rule.csv`,
`submission_r50_d4.csv`). **Submit edildi: public 0.38 (ansambl+rule) və 0.36 (R50) — yeni ən yaxşı rəsmi bal 0.38, yer 310/1191.**

**Run 3 (`filament-unet-r3`, başladı 2026-10-26 ~12:30, ~5.4 saat):**
- `solar`: R34, v4-ün günəş ön-təlimli çəkiləri ilə (`jannatsamadov/filament-solar-pretrain`
  private dataset-i; 276/278 tensor yüklənir, yalnız başlıq təzədir), validasiyalı.
  Müqayisə: ImageNet-li R34 `full` — val 0.4189 / D4 0.4259. Köhnə dəlil: v4 (günəş) rəsmi
  0.27, v2 (ImageNet) 0.29.
- `r50all`: R50, **bütün 707 foto**, 45 epoxa, validasiya yoxdur; post-processing run 2-nin
  R50-dən (D4: 0.75 / 0 / 350). Xəritələri yalnız test üçündür (`val_maps: false`).
- Bitəndə: `aug_post` kernel-inin `kernel_sources`-una `jannatsamadov/filament-unet-r3`
  əlavə et, `python kernels/build.py`, push. Post `solar`-ı skorlayır, `r50all`-u test-only
  kimi göstərir. Yekun ansambl (məs. `r50all` + ən yaxşı validasiyalı R34) test xəritələrində
  qurulmalıdır — bu hələ yazılmayıb.

**Run 3 nəticəsi (bitdi 12:48 UTC; GPU kvotası sonra 15.4 saat qalıb).** Plain val, kernel-in öz grid-i:

| arm | ən yaxşı val PQ | PQ60 | son 4 val ortası | ep 3 |
|---|---|---|---|---|
| `solar` (R34, günəş init) | 0.4172 (ep 54, 0.7/7/500) | 0.3843 | 0.4160 | 0.326 |
| r1 `full` (R34, ImageNet) | 0.4189 (ep 45) | 0.3829 | 0.4156 | 0.359 |
| r2 `r50` (R50, ImageNet) | 0.4248 (ep 42) | 0.3852 | 0.4218 | 0.362 |

- Günəş ön-təlimi heç nə vermir: son nəticə ImageNet ilə eynidir, başlanğıcda isə
  daha zəifdir. **Ölü istiqamət.** `solar`-ın D4 balı hələ yoxdur (post kernel).
- `r50all`: val yoxdur. Kernel-in yazdığı `submission_r50all.csv` **plain** proqnozdur,
  amma 0.75/0/350 D4 xəritələri üçün seçilib. D4 CSV `aug_post`-dan gəlməlidir.
- `kernels/unet_r3/out_kaggle/submission_{solar,r50all}.csv` (1193 və 1385 sətir)
  `verify_submission.py`-dan keçir (3 boş şəkil xəbərdarlığı). **`solar` (plain) submit
  edildi: public 0.35** — R50+D4 (0.36) və ansambldan (0.38) aşağı. `r50all` submit edilməyib.

**Növbəti ideya (istifadəçidən):** şübhəli komponentləri (orta ehtimal) tam həlledicilikli
crop-da ikinci CNN ilə yenidən yoxlamaq — öyrədilmiş keep-rule-un şəkil görən versiyası.

⚠️ `filament-aug-unet`-ə yeni versiya **push etmə** — dump onun son versiyasının
çıxışını oxuyur. ⚠️ Lokal CPU ağır yükdə təsadüfi çökür (yaddaş qeydi
`local-cpu-instability`) — ağır hesablama yalnız Kaggle-da.

## 1. Bir abzasda

GONG Hα tam-disk şəkillərində (2048×2048) günəş filamentlərinin instance
seqmentasiyası. Metrik **Panoptic Quality**. Deadline **15 Noyabr 2026**.
Ən yaxşı rəsmi bal **0.29**, lider **0.54**, dəstə 0.36–0.40.
İştirakçının hədəfi: **0.6+**.

---

## 2. v5 BİTDİ və UĞURSUZ OLDU (2026-08-24)

Təlim epoxa 33-də dayandı. GPU boşdur. **Nəticə mənfidir və qətidir:**

| epoxa | PQ | SQ | RQ | FP | FN |
|---|---|---|---|---|---|
| 3 | **0.3271** | 0.639 | 0.512 | 235 | 497 |
| 15 | 0.2990 | 0.643 | 0.465 | 168 | 563 |
| 30 | 0.2418 | 0.635 | 0.381 | 115 | 647 |

Monoton düşüş. SQ sabit — formalar yaxşı idi. RQ çökdü: model qorxaqlaşdı,
əmin olmadığını proqnoz etməyi dayandırdı. **Ən yaxşı nöqtə self-training
başlamazdan əvvəl idi.** Query head heç öyrənmədi (maksimum PQ 0.069).

⚠️ Bu bal 60 **fotoşəkil** üzərindədir (881 GT); v4-ün 0.3495-i 60 **annotasiya
dəsti** üzərindədir (466 GT). **Müqayisə edilə bilməzlər.** Rəsmi ən yaxşı bal
hələ də **v2 = 0.29**.

**Ayrılmamış:** sayğac etiketləri və self-training eyni qaçışda idi — hansının
çökdüyü bilinmir. Növbəti təcrübə: sayğac etiketləri, self-training **olmadan**.

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
| 7 | v5 (sayğac etiketləri + self-training) | 0.3271→**0.2418** düşdü | — ⚠️ ölü |

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

1. **Sayğac etiketləri, self-training olmadan** — v5-də hansının çökdüyünü ayırmaq
   üçün yeganə yol (§2)
2. **YOLOv8-seg** müstəqil yol kimi (kod yoxdur, 2–3 gün iş). RQ 0.94 hədəfinə
   yeganə açıq istiqamət — anchor-free olduğu üçün Mask R-CNN-in anchor problemi
   ona aid deyil
3. **GONG qonşu günləri + sürüşdürmə ilə etiketləmək** (istifadəçi ideyası,
   `IDEAS.md`). Formanı köçürmək ölüdür (v3); yalnız **toxum kimi** işlədib formanı
   yeni kadrdan çıxarmaq sınanmayıb. Əvvəlcə GPU-suz ölçmə: hər iki günü
   annotasiyalı 582 cütdə proyeksiya→yapışdır→seqment nəticəsinin IoU>0.5 payı
4. Sınanmamış: dərin qatda attention; barb başlığı; xirallik başlığı
5. Submit lazım olsa: **v2** (`spine_v2_best.pt`, §5) — hələ də ən yaxşısıdır

## 9b. Konfrans (2026-09-08/09)

- Tezis göndərilib: `publication/Abstract_Samadov_Dzhalilov_EN.docx`
- Slaydlar: `publication/slides/index.html` (reveal.js, hər şey lokal vendored).
  PDF üçün `index.html?print-pdf` → Ctrl+P → yataq A4, kənarlar yox.
- Slayd şəkilləri: `src/slide_figures.py` (v2 modeli ilə, ~3 dəq GPU)

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
