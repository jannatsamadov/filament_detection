# İdeyalar — sınanmış, sınanmamış, ölü

> Hər ideya bir sətirdə: **kimin**, **nə**, **hansı ölçmə**, **nəticə**.
> Sınanmamış ideya "sınanmayıb" yazılır — "yəqin işləməz" yazılmır.
> Xronoloji hekayə `WALKTHROUGH.md`-də, texniki istinad `izah.md`-də.

---

## 1. Zamanla dəyişmə — model özü fırlanmanı anlasın

**Kim:** iştirakçı
**Nə:** modelə eyni bölgənin bir neçə günlük şəkilləri verilsin, fırlanmanı özü öyrənsin.

**Vəziyyət: SINANMAYIB.**

**Tənqid.** Bu, hazırkı modeldən struktur olaraq fərqlidir — indi model **bir** şəkil görür, burada **bir neçə** görməli olur. Bu, yeni giriş formatı, yeni data yükləyici və 2–3 dəfə çox yaddaş deməkdir (2048×2048, 6 GB kart).

İki əsl problem:
- **Ardıcıl günlərin şəkilləri həmişə yoxdur.** 707 şəkildən yalnız bir hissəsi 1–3 gün aralı cütlər təşkil edir (582 cüt, 3.5 günə qədər).
- **Test zamanı yoxdur.** Qaydaya görə inference **yalnız `test/` qovluğundakı şəkillərlə** olmalıdır. Test şəkilləri arasında ardıcıl gün cütləri varmı — **yoxlanmayıb**. Yoxdursa, model təlimdə öyrəndiyi şeyi testdə istifadə edə bilməz.

**Əvvəl yoxlanmalı:** test dəstində 3.5 gün içində neçə cüt var. Sıfırdırsa, ideya təlim üçün faydalı ola bilər (regularizasiya kimi), amma inference-də işləməz.


add
Sizin təklifin adı var və işləyir — bu, self-training / EM sxemidir. Model öz təxminini etiketə çevirir, təlim davam edir, etiket yenilənir.


---

## 2. Stansiya adını modelə giriş kimi vermək

**Kim:** iştirakçı
**Nə:** altı teleskop var (Bh, Ch, Lh, Mh, Th, Uh). Şəkillərin keyfiyyəti və baxış bucağı fərqlidir. Model bunu bilsin.

**Vəziyyət: SINANMAYIB.**

**Tənqid.** Stansiya kodu **şəklin adındadır, içində deyil** — model onu heç vaxt görmür. Vermək texniki cəhətdən asandır: sabit kanal, ya da 6 ölçülü one-hot.

Baxış bucağı arqumenti **ölçülüb və zəifdir**: stansiyalar arası parallaks **8.3 piksel** (diskin 0.92%-i). Bu, 8 piksel enində filament üçün böyük görünə bilər, amma bütün diski bərabər sürüşdürür — formanı dəyişmir.

Şəkil keyfiyyəti arqumenti isə **ölçülməyib** və güclü ola bilər: fərqli teleskop, fərqli seeing, fərqli fokus.

**Risk:** 566 şəkil azdır. Model "Uh stansiyasında filament az olur" kimi yalançı qayda öyrənə bilər — bu, təlimdə balı qaldırır, testdə salır.

**Əvvəl yoxlanmalı:** stansiyalar arasında ölçülə bilən fərq varmı — kontrast, kəskinlik, küy, disk radiusu, filament sayı. Fərq yoxdursa, əlavə etmək yalnız zərərdir.

---

## 3. Fizika sabitlərini modelə vermək

**Kim:** iştirakçı
**Nə:** Günəşin fırlanma sürəti, Yerin fırlanması, Günəşin radiusu, teleskopun yeri — hamısı giriş kimi.

**Vəziyyət: QISMƏN EDİLİB.**

**Nə edilib.** Modelin 5 fizika kanalı var: `coherence` (lif istiqamətinin nizamı), `sin2t`, `cos2t` (istiqamət), `sinlat` (enlik), `plagedist` (parlaq bölgəyə məsafə). Bunlar **hər piksel üçün dəyişir** və işləyir.

**Tənqid.** İştirakçının saydıqları isə **hər şəkil üçün bir ədəddir**:
- Günəşin fırlanma sürəti — **sabitdir**, dəyişmir. Modelə vermək məlumat vermir.
- Günəşin radiusu — piksel olaraq praktiki sabitdir (904 px, ±10 px). Məlumat yoxdur.
- Yerin fırlanması — şəklin çəkildiyi **saatı** müəyyən edir, diskin görünüşünü yox.
- Teleskopun yeri — **ideya 2**-yə düşür.

**Faydalı ola bilən yeganələr** bunlardır və onlar bu siyahıda yoxdur:
- **B0** — Günəşin oxunun bizə tərəf əyilməsi, ilboyu ±7.2° dəyişir. Yüksək enlikli filamentlərin nə qədər qısaldığını təyin edir. **Real məlumatdır.**
- **P** — ox bucağı, ±26° dəyişir. Şəkildə şimalın harada olduğunu təyin edir. Filamentlərin istiqamət statistikası buna bağlıdır. **Real məlumatdır.**
- **İlin günü / Günəş dövrü fazası** — filament sayı və enlik paylanması 11 illik dövrlə dəyişir. **Real məlumatdır, amma dolayıdır.**

**Sabit olan bir kəmiyyəti modelə vermək heç nə öyrətmir** — bütün nümunələrdə eyni olan giriş, çəkiləri sıfıra gedən girişdir.

---

## 4. "Filament var?" sualına modelin öz cavabı ilə baxmaq

**Kim:** iştirakçı
**Nə:** dünənki filamenti bugünkü şəklə proyeksiya et. O yerdə **model** "filament" deyirsə, amma insan işarələməyibsə → itkin annotasiyadır.

**Vəziyyət: QİSMƏN SINANDI — siqnal TƏSDİQLƏNDİ, `src/check_votes.py`.**

Səs sayları ilə variantı işlətdim (model olmadan, yalnız annotasiyalar). Dünənki
**3/3** filamenti bugünkü şəklə proyeksiya etdim, hara düşdüyünə baxdım.
Dözümlülük 25 piksel — proyeksiya median 14 px yanılır, filament 8 px enindədir,
ona görə "konturun **içinə** düşsün" tələbi həndəsənin verə bilmədiyi dəqiqlikdir.

| Düşdüyü yer | Əsl | Təsadüfi əyri | Nisbət |
|---|---|---|---|
| 3/3 filament | 17.8% | 6.7% | **2.7×** |
| 2/3 | 7.0% | 4.2% | 1.7× |
| **1/3 şübhəli** | **6.4%** | **2.5%** | **2.5×** |
| heç nə | 68.8% | 86.6% | — |

**Siqnal realdır.** Amma məhsul azdır: 21 namizəd, ~10-u təsadüf → **11 etiket**.
Səbəb: üç annotatorlu 151 şəkildən 3.5 gün içində cəmi **19 cüt** çıxır.

**Bunun əsas nəticəsi:** məhdudiyyət ideyada deyil, üç annotator şərtindədir.

### 4b. Səsi günlər arasında saymaq — iştirakçının düzəlişi

**Kim:** iştirakçı. **Vəziyyət: SINANDI VƏ İŞLƏYİR** (`src/check_crossday_votes.py`).

İştirakçı göstərdi ki səsi bir şəklin daxilində saymaq səhvdir: eyni filamenti
üç gündə üç fərqli müəllif işarələyə bilər, mənim kodum bunu üç dəfə "1/3" kimi
görürdü. Qruplar (müəllif komandaları) yoxlandı: qrup daxilində 3.5 gün içində
cəmi **32 cüt**, bütün şəkillər arasında **582 cüt** — və 550-si fərqli
komandalardır, yəni müstəqil şahidlik.

Bütün cütlərdə filamentlər fırlanma ilə bağlandı:

| | Say |
|---|---|
| Proyeksiya | 10303 |
| Annotasiyanın üstünə düşdü | 2867 (**27.8%**) |
| Təsadüfi əyri (nəzarət) | 705 (**8.9%**) |
| Nisbət | **3.1×** |

- **888** filament: şəkildə bir neçə müəllif var, yalnız biri çəkib → **254-ü
  (28.6%)** başqa şəkildə təsdiqlənir.
- **2638** filament: şəkli yalnız bir nəfər işarələyib → **1066-sı (40.4%)**
  başqa şəkildə təsdiqlənir.

**Cəmi 1320 filament (datanın 16%-i)** "şübhəli"dən "təsdiqlənmiş"ə keçə bilər.
Təsadüf payı çıxılsa ~900 real qalır. **Tətbiq ediləcək.**

### 4c. Zəif işarələnmiş filamenti tapmaq balı qaldırırmı?

**Vəziyyət: ÖLÇÜLDÜ** (`src/check_single_vote_value.py`, 60 validasiya dəsti).

| İş nöqtəsi | Olduğu kimi | Tək-müəllif tapıntıları silinsə | Fərq |
|---|---|---|---|
| Sərt (0.90 / 700) | 0.3495 | 0.3311 | **−0.0185** |
| Boş (0.85 / 300) | 0.3301 | 0.3041 | **−0.0260** |

**Silmək zərər verir** → onları tapmaq dəyərlidir. Kalibrləmənin sərtliyi
tək-müəllif filamentlərini deyil, **zibili** kəsir. İkisi fərqli şeydir.

**Niyə perspektivlidir.** Alternativ meyarlar sınandı və uğursuz oldu:

| Meyar | Nəticə |
|---|---|
| Formanı olduğu kimi köçürmək (v3) | 87% qaranlıq olmayan yerə düşür → rəsmi bal 0.29→0.28 |
| Qaranlıqlıq süzgəci | təsadüfi əyri 67%, əsl proyeksiya 76% → **ayırd etmir** |

Model isə qaranlıqlıqdan **daha yaxşı** filament detektorudur — konsensus filamentlərinin 71%-ini tapır.

**Tənqid.** Dairəvi məntiq riski var: model artıq təlim datasında bu filamentləri görüb. Ona görə ölçmə **validasiya bölməsində** aparılmalıdır — model o şəkilləri görməyib.

**Nəzarət məcburidir:** eyni proyeksiyanı təsadüfi yerə atıb modelin cavabına bax. Fərq yoxdursa, siqnal yoxdur.

**İştirakçının tərifi:** "filament = müəlliflərin annotasiya etdiyi şey". Bu, ölçməni birbaşa edir — **3/3 səs almış filament, 1/3 səs almışın üstünə düşürmü?**

---

## Sınanmış və ölmüş

| İdeya | Ölçmə | Nəticə |
|---|---|---|
| Zaman yayılması — formanı köçürmək (v3) | 87% qaranlıq olmayan material | rəsmi 0.29 → **0.28** |
| Qaranlıqlıq meyarı ilə "bura bax" | nəzarət 67% vs əsl 76% (±20 px), 81% vs 91% (±72 px) | **ayırd etmir** |
| Efemeridi düzəltmək | xəta artıq **0.9°** (11–14 px) | **düzəldiləcək bir şey yoxdur** |
| Fırlanma qanununu annotasiyalardan ölçmək | enlik dəyişməsi 3.3–3.6° | **filamentləri tanıya bilmirik**, ölçmə mümkün deyil |
| Səs promote etmək (yalnız annotasiyalarla) | 2.5× təsadüfdən yuxarı, amma cəmi 11 etiket | **siqnal var, həcm yoxdur** |
| Instance etibar balı (əl ilə düzəldilmiş əlamətlər) | 0.3367 → 0.3415, küy 0.03 | **sıfır qazanc** |
| Günəş şəkillərində ön-təlim (v4) | validasiya 0.355 → 0.379 | rəsmi **0.29 → 0.27** |
| **Self-training — model öz etiketlərinə səs versin (v5)** | 33 epoxa, 60 şəkil | **0.3271 → 0.2418, monoton düşüş** |
| **Query head (Mask2Former tipli, Hungarian matching)** | 12 epoxa | PQ **0.069 maksimum**, FP 668 — öyrənmədi |

### v5 self-training niyə öldü (2026-08-24, təlim e33-də dayandı)

| epoxa | PQ | SQ | RQ | FP | FN |
|---|---|---|---|---|---|
| 3 | **0.3271** | 0.639 | 0.512 | 235 | 497 |
| 15 | 0.2990 | 0.643 | 0.465 | 168 | 563 |
| 30 | 0.2418 | 0.635 | 0.381 | 115 | 647 |

**SQ heç tərpənmədi** (0.64) — formalar həmişə yaxşı idi. **RQ çökdü.** FP azaldı,
FN artdı: model getdikcə daha **qorxaq** oldu və əmin olmadığı heç nəyi proqnoz
etməməyə başladı. Etiket sayğacında 0.75-i keçən filament sayı 2946 → 4616
qalxsa da, model daha **az** tapdı.

**Səbəb:** self-training modelin mövcud meylini gücləndirir. Bu qədər qeyri-müəyyən
etiketlə həmin meyl **susmağa** doğrudur. Ən yaxşı nöqtə epoxa 3 idi — yəni
self-training başlamazdan əvvəl.

**Ayrılmamış sual:** etiket etibarlılığı sxemi (sayğac, 0.40/0.50/0.66/0.80/0.90/1.00)
ilə self-training döngəsi eyni qaçışda idi. Çökən hansıdır — bilmirik. Növbəti
təcrübə: sayğac etiketləri ilə **self-training olmadan** öyrətmək.

**Ən vacib dərs:** son üç submitdə **validasiya qalxdı, rəsmi bal düşdü**. 25 şəkillik validasiya artıq etibarsızdır və ona görə optimallaşdırmaq zərərlidir.

**2026-08-22-də ölçüldü:** eyni model, eyni iş nöqtəsi, yalnız validasiya ölçüsü
fərqli — **25 dəst 0.3790, 60 dəst 0.3495**. Fərq **0.03**. v4-ün "qazancının"
böyük hissəsi bu idi.

| Model | Validasiya | Rəsmi | Fərq |
|---|---|---|---|
| v2 | 0.3553 | **0.29** | 0.065 |
| v3 | 0.3593 | 0.28 | 0.079 |
| v4 | 0.3790 | 0.27 | **0.109** |

---

## Sınanmamış, qeyd üçün

### GONG arxivindən qonşu günləri çəkib sürüşdürmə ilə etiketləmək (2026-09-07)

**İdeya (istifadəçidən).** Annotasiya olunmuş şəkillərin yaxın tarixlərindəki
GONG kadrlarını arxivdən çəkmək, differensial fırlanma ilə annotasiyaları o
günlərə proyeksiya etmək və beləcə etiketsiz kadrları da etiketləmək — data
augmentation kimi.

**Qaydaya uyğundur.** `general.md:72`: xarici **data** olar, xarici **ground
truth** olmaz. Burada etiketlər bizim öz MAGFiLO annotasiyalarımızdan həndəsə
ilə çıxarılır, kənardan GT gətirilmir. `src/fetch_gong.py` artıq yazılıb və
işləyir (3772 kadr onunla çəkilib), yəni infrastruktur hazırdır, GPU lazım deyil.

**İdeyanı iki hissəyə ayırmaq lazımdır — biri ölüdür, biri deyil.**

**A. Formanı olduğu kimi köçürmək — ÖLÜ, artıq sınanıb.** Bu, v3-ün özüdür.
Proyeksiya xətası **11–14 px**, filament eni **8.2 px** → köçürülmüş poliqon
həqiqi filamentlə demək olar **sıfır** örtüşür, IoU 0.5-i keçmir, PQ-da həm FN
həm FP sayılır. Ölçüldü: köçürülmüş konturların **87%-i qaranlıq olmayan**
piksellərə düşür, rəsmi bal 0.29 → 0.28. Belə etiketlərlə öyrətmək modelə
filamentin **olduğu yerdən 14 px kənarda** olduğunu öyrədir — faydasız yox,
zərərlidir.

**B. Proyeksiyanı yalnız *toxum* kimi işlətmək, formanı yeni günün şəklindən
çıxarmaq — SINANMAMIŞ, yeganə açıq yol.** Yəni: spine-i proyeksiya et → ±20 px
lokal axtarışla ən qaranlıq silsiləyə "yapışdır" → maskanı yapışdırılmış
spine-dan yeni kadrın öz parlaqlığı ilə böyüt. Forma nə dünənin poliqonundan,
nə modelin öz təxminindən gəlir — deməli nə v3-ün, nə v5 self-training-in
uğursuzluğunu miras almır.

**Diqqət — burada incə fərq var.** `check_lookhere.py` qaranlıqlıq meyarını
sınadı və **ayırd etmədi** (nəzarət 67% vs əsl 76%, ±20 px). Amma o test
başqa sual verirdi: *"burada ümumiyyətlə filament varmı?"* Bu ideya isə
soruşur: *"filamentin burada olduğunu bilirik (dünən ekspert çəkib) — dəqiq
harada?"* **Detektor kimi işləməyən meyar lokalizator kimi işləyə bilər.**
Bu ayrı sualdır və ölçülməyib.

**Qərar verən təcrübə — GPU lazım deyil, bir neçə saat iş.** Hər iki günü
annotasiyalı olan cütlərdə (582 cüt var, 27.8% əlaqələnir):
1. 1-ci günün annotasiyasını 2-ci günə proyeksiya et,
2. ±20 px lokal axtarışla ən qaranlıq silsiləyə yapışdır,
3. yenidən seqmentləşdir,
4. nəticəni 2-ci günün **həqiqi** annotasiyası ilə müqayisə et — IoU > 0.5
   payı nədir?

Əgər bu pay aşağıdırsa ideya ölüdür və bunu **heç nə öyrətmədən** bilirik.
Yüksəkdirsə, metod istifadə oluna bilən etiket istehsal edir.

**Ən ciddi etiraz — seçmə meyli (selection bias).** Filamentin proyeksiya
oluna bilməsi üçün o, **növbəti günə qalmalı və çəkilmiş olmalıdır**. Uzunömürlü,
iri, aydın filamentlər — yəni modelin **artıq tapdıqları** (konsensus recall
**71%**). Qısaömürlü, sönük, mübahisəli olanlar — yəni **buraxdıqlarımız**
(tək annotator recall **29%**) — məhz proyeksiya olunmayanlardır.

Deməli bu augmentation **asan nümunələri çoxaldır, çətinləri yox.** Bizim
itkimiz SQ-də (0.64, formalar yaxşıdır) deyil, RQ-dədir (0.45, tapa bilmirik) —
və tapa bilmədiklərimiz tam olaraq bu metodun gətirə bilmədikləridir.

**Nəticə:** B variantı ucuzdur və sınanmalıdır, amma gözlənti təvazökar
olmalıdır. Yuxarıdakı 4 addımlıq ölçmə hər şeydən əvvəl edilməlidir —
əgər IoU payı aşağıdırsa, kadr çəkməyə də dəyməz.

---

- **YOLOv8-seg.** Mask R-CNN uğursuzluğuna əsaslanaraq kənara qoyulmuşdu. O əsas
  **yanlışdır**: YOLO-seg anchor-free işləyir, qutu-IoU uyğunlaşmasından asılı
  deyil. Liderlərdən birinin açıq notebook-u YOLOv8x-seg, 1792 px göstərir.

  **2026-08-22:** "YOLO-nun üstünlüyü instance etibar balındadır, onu öz boru
  xəttimizə ucuz yolla əlavə edə bilərik" fərziyyəsi **sınandı və rədd edildi**
  (`src/check_instance_score.py`). Doqquz əlamət üzərində loqistik model,
  validasiyanın yarısında öyrədilib o biri yarısında ölçüldü: 0.3367 → 0.3415,
  yəni küy içində. Boş həddə çıxan **699 namizədin yalnız 39.5%-i realdır** və
  bu əlamətlər onları ayıra bilmir. Deməli problem instance **seçimində** deyil,
  namizədlərin özündədir — yəni təlimdə. YOLO-ya keçməyin ucuz əvəzi yoxdur.

- **U-Net-in dərin qatına attention.** Filament uzun və nazikdir, uzaq kontekst
  vacibdir. Tam transformer bu kartda mümkün deyil (2048 px-də 16384 token,
  attention kvadratik) və 566 şəkil transformer üçün azdır. Amma ən dərin qatda
  (64×64 = 4096 token) bir attention bloku ucuzdur. Sınanmayıb.

- **Tam kadrla təlim.** Ölçüldü: 2048 px, batch 1, **4749 MiB** — 6 GB kartda
  sığır. Kəsiklərlə təlimin gizli qüsuru var: model "filament burada bitir"
  siqnalını süni kəsik kənarlarından öyrənir, testdə isə o kənarlar yoxdur.
  Qiyməti: epoxa başına 2.7× hesablama (31 dəq → ~85 dəq).
- **Validasiyanı düzəltmək.** 25 şəkil → ən azı 60. **Ölçüldü: fərq 0.03.** Bütün
  qərarlar bundan sonra 60 dəstlə verilir.
- **Xirallik başlığı.** `category_id` istifadə edilmir: Left 2535, Right 2590,
  **Unidentifiable 3074**. Metrikə birbaşa təsir etmir (tək sinif), amma əlavə
  nəzarət tapşırığı kimi əsas tapşırığı gücləndirə bilər. Üstəlik
  "Unidentifiable" hazır **zəiflik göstəricisidir**. Ucuzdur: bir 4-sinifli
  çıxış başlığı.
- **Parlaqlıq eyniləşdirməsini çıxarmaq.** İştirakçının təklifi. Ölçülüb ki o addım kontrastı 0.667σ → 1.331σ qaldırır, amma ön-təlim də, fizika kanalları da onun üzərində qurulub — çıxarmaq hamısını yenidən qurmaq deməkdir.
- **Barb-ları ayrıca başlıqla öyrətmək.** Filamentlərin 81.5%-də barb var, median sahənin 18%-i.
