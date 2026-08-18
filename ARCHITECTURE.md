# Modelin quruluşu — sözlə və sxemlə

> Hazırkı işləyən sistem (v2, rəsmi bal 0.29). Hər qərarın arxasındakı ölçmə
> `WALKTHROUGH.md`-dədir; bu fayl **necə işlədiyini** izah edir.
>
> Son yenilənmə: 2026-08-18

---

## 1. Bir baxışda

```
                    ┌─────────────────────────────────────────┐
   GONG Hα JPEG     │            HAZIRLIQ (bir dəfə)          │
   2048×2048  ─────▶│                                         │
                    │  disk tap ──▶ limb sönməsini böl        │
                    │  (radial     (hər şəklin öz profili)    │
                    │   törəmə)                               │
                    │       │                                 │
                    │       └──▶ 5 fizika kanalı hesabla      │
                    └───────────────────┬─────────────────────┘
                                        │  6 kanal, 2048×2048
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │              U-Net (ResNet-34)          │
                    │           512×512 patch, 24.45 M        │
                    └───────────────────┬─────────────────────┘
                                        │
                      ┌─────────────────┼─────────────────┐
                      ▼                 ▼                 ▼
                 ┌─────────┐      ┌──────────┐      ┌──────────┐
                 │ semantik│      │  spine   │      │  radius  │
                 │filament?│      │mərkəz xət│      │ yarımen  │
                 └────┬────┘      └─────┬────┘      └─────┬────┘
                      │                 │                 │
                      │                 ▼                 │
                      │        həddi keç (0.95)           │
                      │                 │                 │
                      │                 ▼                 │
                      │      connected components         │
                      │      = hər biri bir filament      │
                      │                 │                 │
                      │                 ▼                 │
                      │      nöqtələri sırala + splayn     │
                      │                 │                 │
                      │                 └────────┬────────┘
                      │                          ▼
                      │              əyri boyunca disk sürüşdür
                      │                          │
                      └──────────► kəsişmə ◄─────┘
                                     │
                                     ▼
                          instance maskaları ──▶ RLE ──▶ CSV
```

---

## 2. Niyə belə — qısa əsaslandırma

Filament **median 8.2 piksel enində, 125 piksel uzunluğunda əyri lentdir** və öz
düzbucaqlı qutusunun cəmi **24%-ni** doldurur.

Bu, qutu-əsaslı detektorları (Mask R-CNN, YOLO) sıradan çıxarır: 2665 qutu üzərində
ölçdük ki, standart FPN anchor-larının **yalnız 33%-i** IoU 0.7-yə çata bilir.
Mask R-CNN üç konfiqurasiyada da çökdü, PQ 0.000.

Ona görə **qutu yoxdur, anchor yoxdur, NMS yoxdur**. Instance-lar həndəsi qurulur.

---

## 3. Addım-addım

### 3.1 Hazırlıq — disk tapılması

Limb **radial törəmənin ən dik mənfi nöqtəsidir** (Hα-da limb parlaqlaşması kəskin
sıçrayış verir, ondan sonra şəkildəki ən böyük eniş gəlir). 720 şüa atılır, dairə
fit edilir, kənar nöqtələr **median ilə** süzülür.

⚠️ Sadəlövh üsul (həd + sentroid) radiusu **13% böyük** verir — diskin ətrafındakı
səpilmiş işıq halqasını Günəş sayır.

Nəticə sabitdir: mərkəz (1023, 1024) ± 0.5 px, radius **904 ± 0.8 px**.

### 3.2 Hazırlıq — fotometrik düzəliş

Şəkillər həndəsi olaraq oturdulub, **parlaqlıq isə yox**: `parlaqlıq(0.9R)/parlaqlıq(0.1R)`
şəkildən şəkilə **0.719 ↔ 0.849** oynayır (atmosferdən, Günəşdən deyil).

Hər şəklin **öz** sakit-Günəş radial profili çıxarılır (75-ci persentil ilə —
median filamentlərin özü tərəfindən aşağı dartılır) və bölünür.

| | xam | düzəldilmiş |
|---|---|---|
| Filament kontrastı | 0.667 σ | **1.331 σ** |
| Fonla qarışıqlıq | 27.9% | **11.4%** |

Bu addım **hər şeyin əsasıdır** — kontrastı iki dəfə artırır.

### 3.3 Fizika kanalları

Modelə şəkildən başqa beş kanal verilir, hamısı elə həmin Hα şəklindən çıxarılır:

| Kanal | Nə deyir | Ölçülmüş əsas |
|---|---|---|
| **koherentlik** | fibrillərin nə qədər düzülü olması | maska içi 0.55, halqa 0.375, uzaq fon 0.282 |
| **sin2θ, cos2θ** | fibril oxunun heliqrafik E-W-yə nisbətən bucağı | fibrillər filament oxundan 15.6° (təsadüfi 45°) |
| **sin(enlik)** | heliqrafik enlik | oriyentasiya enlikdən asılıdır, p = 8.6×10⁻⁴⁴ |
| **plage məsafəsi** | aktiv bölgəyə yaxınlıq | iki filament populyasiyasını ayırır, sunspot-ları işarələyir |

Çəkiləri **sıfırdan** başlayır — model onları istifadə etməyə məcbur deyil. Təlim
sonunda birinci qatın çəki enerjisinin **12.5%-i** onlardadır, ən çoxu koherentlikdə.

### 3.4 Model

**U-Net**, ResNet-34 encoder (ImageNet), klassik decoder + skip bağlantıları,
24.45 M parametr. Giriş 6 kanal, çıxış üç başlıq.

Tam **2048 ayırdetmədə**, 512×512 patch-lə. (1024-də təlim səhv idi: filament eni
8 px-dən 4 px-ə düşür, barblar itir.)

**Augmentasiya yalnız güzgülərlə.** Enlik giriş kanalı olduğu üçün məsələ artıq
fırlanmaya simmetrik deyil — 90° fırlanmış Günəş mövcud olmayan obyektdir.
Güzgülərdə `sin2t` və `sinlat` kanallarının işarəsi analitik düzəldilir.

### 3.5 Hədəflər — konsensus

Hər **şəkil** bir nümunədir (annotator başına deyil), hədəf isə **"neçə annotator
bu pikseli işarələdi"** — 1.0 / 0.67 / 0.33.

Səbəb: əvvəl hər annotatorun versiyası ayrıca nümunə idi, ona görə mübahisəli
filament modelə **bir dəfə pozitiv, iki dəfə fon** kimi göstərilirdi. Model
çoxluğun tərəfini tuturdu: konsensus filamentlərinin 71%-ini tapırdı, tək
annotatorun gördüklərinin isə **29%-ini**.

Üç annotatorlu şəkillərdə işarələnmiş piksellərin **üçdə biri, bəzən yarısı**
yalnız bir səs daşıyır.

### 3.6 Instance yığması

```
spine xəritəsi ──▶ həddi keç ──▶ connected components
                                        │
                            hər komponent = bir filament
                                        │
                     nöqtələri əsas ox boyunca sırala
                     (≤400-ə seyrəlt, sonra greedy NN)
                                        │
                          hamarlayıcı splayn fit et
                                        │
                     boyunca radius xəritəsindən disk sürüşdür
                                        │
                        semantik maska ilə kəsişdir
                                        │
                                  instance maskası
```

**Niyə işləyir:** spine-lar maskalardan yaxşı ayrılır — cəmi **0.9%**-i başqa
spine-dan 10 px yaxındır (maskalarda 2.8%). Yəni connected components instance-ları
özü ayırır.

**Nə qazandırır:** hər instance **sürüşdürülmüş əyridir**, yəni konstruksiyaya görə
tək əlaqəli komponentdir. PQ-nun ən sərt cəzalandırdığı **fraqmentasiya riyazi
olaraq mümkün deyil**.

### 3.7 Kalibrləmə və TTA

İki parametr süpürülür: **spine həddi** (0.5–0.95) və **minimal spine uzunluğu**
(300–700 px). Şəbəkə hər kadr üçün bir dəfə işləyir, xəritələr keşlənir.

**TTA:** dörd güzgü variantı, orta. Ən böyük tək qazanc (+0.013).

---

## 4. Boru xəttinin tavanı

Modeli kənara qoyub **həqiqi annotasiyalardan düzəldilmiş mükəmməl xəritələri**
yığma kodundan keçirdik:

| | PQ | SQ | RQ |
|---|---|---|---|
| **Boru xəttinin tavanı** | **0.734** | 0.753 | 0.976 |
| İnsan konsensusu (iki ekspert) | 0.348 | 0.595 | 0.542 |
| **Hazırkı model** | **0.29 (rəsmi)** | 0.64 | 0.55 |

Tavan 1.0 deyil, çünki spine boyunca sürüşdürmə **barbları** təsvir edə bilmir —
maskanın median **18%-i** barbdır (filamentlərin 81.5%-ində var, ölçü ilə artır).

**Darboğaz modeldir**, kod və ya təmsil deyil.

---

## 5. Hazırkı zəiflik

İş nöqtəsində: **TP 117, FP 69, FN 119**.

SQ **0.64** — iki ekspertin bir-biri ilə razılaşmasından (0.595) yüksəkdir. Yəni
model bir filamenti tapdıqda konturu yaxşı çəkir.

RQ **0.55** — insan səviyyəsindən (0.542) az fərqlə yuxarı, amma **filamentlərin
yarısını itiririk**. Aqressiv həddlə FP-ni 444-dən 69-a endirdik, qiyməti recall oldu.

Növbəti qazanc **recall-u FP qaytarmadan qaldırmaqdan** gəlməlidir.

---

## 6. Fayl xəritəsi

| Mərhələ | Fayl |
|---|---|
| disk tapılması | `src/geometry.py` |
| fotometrik düzəliş | `src/photometry.py` |
| fizika kanalları | `src/physics_channels.py` |
| konsensus hədəfləri | `src/precompute_consensus.py` |
| dataset, augmentasiya | `src/spine_data.py` |
| model | `src/spine_model.py` |
| təlim | `src/spine_train2.py` |
| instance yığması | `src/spine_infer.py` |
| kalibrləmə | `src/spine_calibrate.py` |
| test proqnozu | `src/spine_predict.py` |
| submission yoxlaması | `src/verify_submission.py` |
| PQ metriki | `src/metrics.py` — **rəsmi kodla yoxlanılıb, fərq 6×10⁻¹⁰** |
| GPU idarəsi | `src/thermal.py`, ortaq `gpu_guard.py`, `gpu_lock.py` |

Ölçmə skriptləri `src/check_*.py` — hər dizayn qərarının arxasındakı rəqəm.
