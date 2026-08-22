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
Model cavab versə, o şərt qalxır — **19 cüt yox, 582 cüt**. Ona görə modelli
variant indi əsaslandırılmış addımdır, fərziyyə deyil.

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
| Günəş şəkillərində ön-təlim (v4) | validasiya 0.355 → 0.379 | rəsmi **0.29 → 0.27** |

**Ən vacib dərs:** son üç submitdə **validasiya qalxdı, rəsmi bal düşdü**. 25 şəkillik validasiya artıq etibarsızdır və ona görə optimallaşdırmaq zərərlidir.

| Model | Validasiya | Rəsmi | Fərq |
|---|---|---|---|
| v2 | 0.3553 | **0.29** | 0.065 |
| v3 | 0.3593 | 0.28 | 0.079 |
| v4 | 0.3790 | 0.27 | **0.109** |

---

## Sınanmamış, qeyd üçün

- **YOLOv8-seg.** Mask R-CNN uğursuzluğuna əsaslanaraq kənara qoyulmuşdu. O əsas **yanlışdır**: YOLO-seg anchor-free işləyir, qutu-IoU uyğunlaşmasından asılı deyil. Liderlərdən birinin açıq notebook-u YOLOv8x-seg, 1792 px göstərir.
- **Validasiyanı düzəltmək.** 25 şəkil → bütün validasiya bölməsi. GPU vaxtı artır, amma qərarlar etibarlı olur. Bu, hazırda **ən ucuz və ən vacib** düzəlişdir.
- **Parlaqlıq eyniləşdirməsini çıxarmaq.** İştirakçının təklifi. Ölçülüb ki o addım kontrastı 0.667σ → 1.331σ qaldırır, amma ön-təlim də, fizika kanalları da onun üzərində qurulub — çıxarmaq hamısını yenidən qurmaq deməkdir.
- **Barb-ları ayrıca başlıqla öyrətmək.** Filamentlərin 81.5%-də barb var, median sahənin 18%-i.
