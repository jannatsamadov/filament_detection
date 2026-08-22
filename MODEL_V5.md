# v5 — modelin tam ideyası

> Bu sənəd v5-in **nə olduğunu** və **niyə belə olduğunu** saxlayır.
> Ölçmələr `IDEAS.md`-də, boru xəttinin izahı `ARCHITECTURE.md`-də,
> hazırkı vəziyyət `HANDOFF.md`-dədir.
>
> Tarix: 2026-08-23

---

## Niyə v5

v4-də iş dayandı: rəsmi bal **0.27**, ən yaxşımız hələ də **0.29** (v2).
Səbəb ölçüldü — problem instance seçimində deyil, **modelin təklif etdiyi
namizədlərdədir**. Boş həddə çıxan 699 namizədin yalnız **39.5%-i** realdır və
heç bir sıralama üsulu onları ayıra bilmir (`check_instance_score.py`, qazanc
+0.005, küy 0.03).

Yəni düzəliş **inference-də deyil, təlimdədir**: hədəflər və girişlər.

Hazırkı iş nöqtəsində **TP 249, FP 186, FN 217** — filamentlərin yarısını
tapmırıq. Və ölçüldü ki zəif işarələnmiş filamentləri tapmaq **balı qaldırır**
(onları silmək −0.019 və −0.026 verir). Yəni recall-a getmək düzgün istiqamətdir.

---

## 1. Giriş — 17 kanal, tam 2048 kadr

| Kanal | Say | Niyə |
|---|---|---|
| Düzləşdirilmiş şəkil | 1 | ön-təlim bunun üzərində aparılıb |
| **Xam şəkil** | 1 | iştirakçı: düzləşdirmə keyfiyyəti itirir, qaranlıq zonalar yaradır |
| Fizika: `coherence` | 1 | fibrillərin nizamı (maska içi 15.6°, halqa 23.1°, təsadüfi 45°) |
| Fizika: `sin2t`, `cos2t` | 2 | fibril istiqaməti; iki kanal, çünki istiqamət 180°-də təkrarlanır |
| Fizika: `sinlat` | 1 | pikselin heliqrafik enliyi |
| Fizika: `plagedist` | 1 | ən yaxın parlaq bölgəyə məsafə |
| Stansiya (one-hot) | 6 | altı teleskop, fərqli keyfiyyət |
| B0 | 1 | Günəş oxunun bizə əyilməsi, ±7.2°, qısalmanı təyin edir |
| P | 1 | ox bucağı şəkil müstəvisində, ±26° |
| İlin günü (sin, cos) | 2 | Günəş dövrü fazası; filament sayı və enliyi bununla dəyişir |
| **Cəmi** | **17** | |

**Niyə həm xam, həm düzləşdirilmiş.** İştirakçı düzləşdirməni çıxarmaq istədi
(vizual qərar). Amma ön-təlim (3772 GONG şəkli, 10 saat) düzləşdirilmiş
şəkillərdə aparılıb — yalnız xam versək, o iş qismən boşa gedər. Hər ikisini
kanal kimi vermək seçimi modelə buraxır və heç nə itmir. Qiyməti: bir kanal.

**Niyə tam kadr, kəsik yox.** Kəsiklə təlimin gizli qüsuru var: model
"filament burada bitir" siqnalını **süni kəsik kənarlarından** öyrənir, testdə
isə tam kadr görünür və o kənarlar yoxdur. Ölçüldü: 2048 px batch 1 =
**4749 MiB**, 6 GB kartda sığır.

**Nə verilmir və niyə:**
- Fırlanma sürəti, Günəşin radiusu — **sabitdir**, bütün nümunələrdə eyni giriş
  heç nə öyrətmir.
- Müəllif kimliyi, xirallik — **testdə yoxdur**, giriş ola bilməz.

---

## 2. Çıxışlar — dörd başlıq

| Başlıq | Nə verir | Loss |
|---|---|---|
| semantik | filament maskası | Dice + BCE |
| spine | orta xətt istiliyi | Dice + BCE |
| radius | yarımen | L1, yalnız maska içində |
| **xirallik** (yeni) | Left / Right / Unidentifiable / Ambiguous | cross-entropy, yalnız maska içində |

**Niyə xirallik.** `category_id` sahəsi datada var və indiyə qədər işlədilməyib:
Left 2535, Right 2590, **Unidentifiable 3074**. Metrikə birbaşa təsir etmir
(tək sinif), amma modeli filamentin **daxili quruluşuna** baxmağa məcbur edir.
Əlavə olaraq "Unidentifiable" hazır zəiflik göstəricisidir.

---

## 3. Etiketlər — sayğac sistemi

Hər filament üçün iki ədəd: **lehinə** `a`, **əleyhinə** `b`.
Bal = `a / (a + b)`. **Filament sayılma həddi: 0.75.**

### 3a. Başlanğıc — insan dəlili

| Müəllif sayı | İşarələyən | Bal | a | b |
|---|---|---|---|---|
| 1 | 1 | **0.66** | 3.3 | 1.7 |
| 2 | 1 | **0.50** | 2.5 | 2.5 |
| 2 | 2 | 1.00 | — toxunulmaz |
| 3 | 1 | **0.40** | 2.0 | 3.0 |
| 3 | 2 | **0.80** | 4.0 | 1.0 |
| 3 | 3 | 1.00 | — toxunulmaz |

Başlanğıc çəki hər halda **5**-dir, yəni model səsləri onu tərpədə bilir.
Bütün müəlliflərin çəkdiyi filament **dəyişməzdir** — nə model, nə həndəsə onu
aşağı sala bilər.

### 3b. Çarpaz-gün təsdiqi — müəllif məntiqi ilə

Dünənki filament fırlanma ilə bugünə köçürülür (xəta ölçülüb: **11–14 piksel =
0.9°**). Düşdüyü yerdə annotasiya varsa, bu bir təsdiqdir. Amma **kimin**
təsdiqlədiyi əhəmiyyətlidir:

| Təsdiq növü | Nə qədər | Niyə |
|---|---|---|
| **Fərqli müəllif, fərqli gün** | **əlavə müəllif səsi** kimi | müstəqil şahidlikdir; 3/1 → 3/2 kimi hesablanır |
| **Eyni müəllif, fərqli gün** | `a += 2.5` | fərqli şəkildir, amma eyni adamın gözü — zəifdir, ~2-3 epoxalıq |

Ölçüldü (`check_crossday_votes.py`): proyeksiyaların **27.8%-i** annotasiyaya
düşür, təsadüfi əyri ilə **8.9%** — **3.1×**. Cəmi **1320 filament** çarpaz-gün
dəlili alır (254-ü çox müəllifli şəkillərdən, 1066-sı tək müəllifli).

**Yalnız həndəsə ilə heç nə keçmir.** 0.40-dan başlayan filament bir eyni-müəllif
təsdiqi ilə 0.54-ə qalxır — hədd 0.75-dir.

### 3c. Model səsi — self-training

Hər epoxada model təlim kadrlarına baxır və hər namizədə səs verir:
ehtimal > 0.5 → `a += 1`, əks halda `b += 1`. **Maksimum 30 səs.**

Yoxlama, iştirakçının nümunəsi ilə — 1/3 filament, 25 təsdiq / 5 rədd:
`a = 2 + 25 = 27`, `b = 3 + 5 = 8` → **0.77 → keçir.**
Əksi (5 təsdiq / 25 rədd): `a = 7`, `b = 28` → **0.20 → atılır.**

**Dairəvi məntiq riski** üç qayda ilə bağlanır:
1. İnsan etiketi **döşəmədir** — model onu aşağı sala bilməz.
2. 3/3 və 2/2 filamentlər **toxunulmazdır**.
3. Model səsi 30 epoxa ilə məhduddur — sonsuz yığılma yoxdur.

### 3d. İnsan heç işarələməyibsə

Ayrı qapı, iki mərhələ:
1. **Yer** — insan işarəsindən gəlir: dünənki filament fırlanma ilə köçürülür.
   Təsadüfi yerə baxılmır.
2. **Varlıq** — model deyir. Başlanğıc `a = 1.5, b = 3.5` (bal 0.30), sonra
   epoxalarda səs yığır.

0.75-i keçmək üçün ~20 epoxa model təsdiqi lazımdır. Bu, v3-dən köklü fərqdir:
v3 proyeksiyanı **birbaşa etiket kimi yapışdırırdı** və 87%-i qaranlıq olmayan
materiala düşürdü.

---

## 4. Təlim cədvəli — 6 / 24 / 6

| Mərhələ | Epoxa | Nə olur |
|---|---|---|
| **İsinmə** | 1–6 | yalnız insan etiketləri. Model hələ səs verməyə layiq deyil |
| **Self-learning** | 7–30 | hər epoxada etiketlər yenilənir |
| **Oturma** | 31–36 | etiketlər **dondurulur**, lr sıfıra enir |

**Niyə isinmə lazımdır.** Model 1-ci epoxada təsadüfi cavab verir; onun səsi
etiketi korlayar.

**Niyə oturma lazımdır.** Etiketlər son epoxaya qədər dəyişsə, model hərəkət
edən hədəfin arxasınca qaçır və çəkilər yarımçıq qalır.

**Niyə 24.** Sayğaca yer lazımdır: 0.40-dan 0.75-ə çatmaq ~20 təsdiq tələb edir.
12 epoxa olsaydı, **hamısı** təsdiq olmalı idi — həddindən artıq sərt.

---

## 5. Təlim parametrləri

| | |
|---|---|
| Başlanğıc çəkilər | `spine_v4_best.pt` (GONG ön-təlimi + fine-tune) |
| Optimizer | AdamW, weight decay 1e-4 |
| lr | **1e-4** (batch 1 üçün azaldılıb) |
| Cədvəl | OneCycle |
| Batch | 1 (tam 2048 kadr) |
| Addım/epoxa | 566 |
| Dəqiqlik | AMP, qradiyent kəsimi 5.0 |
| Augmentasiya | güzgü (fizika kanallarında işarə düzəlişi ilə) |
| Validasiya | **60 dəst**, hər 3 epoxada |
| Kart | 55% hədəf, `gpu_guard` + `gpu_lock` |

**Müddət: epoxa ~85 dəqiqə, 36 epoxa ≈ 51 saat.** Checkpoint hər epoxadan sonra;
dayandırıb davam etmək olar.

---

## 6. Nəyi bilməyəcəyik

Bir anda **altı** şey dəyişir: tam kadr, xam kanal, yeni hədəflər, self-training,
yeni giriş kanalları, xirallik başlığı.

Nəticə pis çıxsa, **hansının günahı olduğunu ayıra bilməyəcəyik.** Ayrı-ayrı
sınamaq 6 gecə tələb edər və validasiya küyümüz (**0.03**) onları onsuz da ayırd
edə bilməzdi — bu, ölçülüb: eyni model 25 dəstdə 0.3790, 60 dəstdə 0.3495.

Ona görə hamısı birdən gedir — bilərək, təsadüfən yox.

---

## 7. Uğur meyarı

**60 dəstlik validasiyada v4-ün 0.3495-ini keçmək.** Keçməsə, submit edilmir.

Rəsmi bal validasiyadan sistematik **0.05–0.09 aşağıdır**, ona görə 0.3495
təxminən 0.27–0.29 rəsmi bala uyğundur. Leaderboard dəstəsi **0.36–0.40**-dır;
oraya çatmaq üçün validasiyada **0.42+** lazımdır.
