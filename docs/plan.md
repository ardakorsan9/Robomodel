# robomodel planı

Saatler tahmindir ve oturumlarda geçen zamanı da kapsar. Bir aşamanın **kabul testleri**
geçmeden sonraki aşamaya geçilmez.

## 0. Kapsam: kesin olanlar ve konuşulacaklar

**Kesin (şimdi yazılıyor):** CAD → tam Fossen modeli → simülasyon → MATLAB zinciri.

| # | Özellik | Neden şart |
|---|---|---|
| K1 | Geometri, rijit gövde, hidrostatik, `Parameter` (değer ± belirsizlik, kaynak) | Modelin temeli (**tamam**) |
| K2 | Şekilli parçalar (kutu, silindir, küre): kendi eylemsizliğiyle | Batarya gibi büyük parçalar nokta kütle sayılırsa eylemsizlik yanlış çıkar |
| K3 | Eklenmiş kütle: elipsoitte Lamb (kesin), diğer her gövdede şerit teorisi + eşdeğer sferoit düzeltmesi | Denklemdeki M_A terimi; her CAD için bir değer üretmek |
| K4 | Coriolis C_RB(ν), C_A(ν) | Denklemin terimi, M'den formülle çıkar |
| K5 | Sönümleme tahmini (doğrusal + karesel, izdüşüm alanları ve C_d) | Denklemin D terimi; geniş belirsizlikle |
| K6 | İticiler: konum, yön, itki sınırı, dağıtım matrisi B | τ = B·u: kontrolcünün giriş matrisi |
| K7 | Python 6-DOF simülatörü + fizik testleri (denge, enerji, son hız, salınım periyodu) | Modelin tutarlı olduğunu kanıtlamak |
| K8 | MATLAB çıktısı: parametreler, dinamik fonksiyonu, örnek `ode45`, doğrusal A, B | Kontrolcü tasarımı (`lqr`, `pidtune`) |

**Konuşulacak (isteğe bağlı):** artı ve eksileri ayrıca konuşulacak.

| # | Özellik | Not |
|---|---|---|
| O1 | Gazebo / URDF çıktısı + eksen dönüşüm modülü | ROS 2 ile kullanmak için |
| O2 | Capytaine (BEM) ile eklenmiş kütle | Kutu gibi, ince olmayan gövdelerde şerit teorisinden çok daha iyi |
| O3 | Havuz verisiyle kalibrasyon | Sönümlemeyi %90+ uyuma çıkarmanın tek yolu |
| O4 | Ölçüm sihirbazı | "Şunu ölç, doğruluk şu kadar artar" |
| O5 | REMUS 100 / BlueROV2 doğrulaması | Yayınlanmış değerlerle kanıt; kaynak verisi gerekli |
| O6 | Kanat ve uzantılar (fin, iticilerin dış hacmi) | Torpido araçlarda yunuslama ve dönme için önemli |
| O7 | Diğer robot kategorileri, STEP okuma, HTML rapor, arayüz | |

## 1. Doğruluk hedefi: neyi %100'e yakın yapabiliriz?

Bir robot modelinin parametreleri üç gruba ayrılır. CAD'den çıkarılabilen doğruluk
gruptan gruba çok değişir.

| Grup | Örnek | CAD'den çıkar mı? | Hedef |
|---|---|---|---|
| **A. Geometri ve kütle** | Hacim, kaldırma merkezi, kütle merkezi, eylemsizlik, itici konumları | Evet, tamamen | **≥ %99**: analitik çözüme karşı test edilir |
| **B. İdeal akış ve temas** | Eklenmiş kütle (potansiyel akış), tekerlek kinematiği, pervane geometrisi | Evet, fizik modeliyle | Analitik gövdelerde ≥ %99, gerçek araçta %80–90 |
| **C. Sürtünme ve sönümleme** | Su sönümlemesi, tekerlek kayması, eklem sürtünmesi, motor sabitleri | **Hayır** | CAD'den kaba tahmin (±%30–100), **kalibrasyonla ≥ %90 uyum** |

"%100'e yakın" hedefi, A ve B gruplarında **analitik test gövdeleri** için geçerli; her
aşamada bunu sayılarla gösteriyoruz. C grubu hiçbir yazılımda sadece CAD'den çıkmaz. Bu
yüzden yazılım her değerin yanında belirsizliğini ve kaynağını yazar. Hangi değerin
kalibrasyon gerektirdiği böylece raporda açıkça görünür.

Gerçek araçta A grubunun doğruluğu CAD'in gerçeğe ne kadar benzediğine bağlıdır.
Batarya ve elektronik CAD'de yoksa kütle merkezi yanlış çıkar. Bunun için YAML'daki
`parts` alanı var.

## 2. Doğrulama merdiveni

Her yeni hesap önce cevabı kesin bilinen bir durumda denenir:

| Basamak | Gövde | Bilinen cevap | Kaynak |
|---|---|---|---|
| 1 | Küre | V, I, eklenmiş kütle = ½ρV | Analitik |
| 2 | Elipsoit / uzun sferoit | Lamb katsayıları (k1, k2, k') | Lamb 1932; Newman, *Marine Hydrodynamics* |
| 3 | Silindir, Myring gövdesi | Hacim ve eylemsizlik (sayısal integral) | Analitik integral |
| 4 | REMUS 100 | Tüm katsayılar | [Prestero 2001, MIT](https://dspace.mit.edu/handle/1721.1/65068) |
| 5 | BlueROV2 | Deneysel katsayılar | [von Benzon vd. 2022](https://mdpi.com/2077-1312/10/12/1898) |
| 6 | Kendi aracımız | Havuz testi | Bizim ölçümümüz |

## 3. Aşamalar: sualtı araçları (MVP)

### Faz 0. Hazırlık: 2–3 saat (**tamam**)
- Repo, uv, pytest, ruff, YAML şeması, `robomodel check`, eksen kuralları
- **Kabul:** örnek dosyalar okunuyor, hatalı dosyada tüm hatalar tek seferde listeleniyor

### Faz 1. Çekirdek: 12–16 saat (**tamam**, eksen modülü O1 ile birlikte)
| İş | Denenecek | Kabul ölçütü |
|---|---|---|
| `Parameter` tipi (değer, belirsizlik, kaynak) | Her çıktı değerinde kaynak ve belirsizlik var | Testte zorunlu |
| Geometri: küre, elipsoit, silindir, Myring, STL | Ağ hacmi ve analitik hacim | Hata < %0,1 |
| Rijit gövde: kütle merkezi, eylemsizlik, M_RB | Küre, silindir, nokta kütleler, paralel eksen teoremi | Hata < %0,1 |
| Hidrostatik: V, kaldırma merkezi, W, B, g(η) | Nötr kaldırma, 90° yuvarlanmada moment | Analitik ile birebir |
| Eksen modülü (NED ↔ ENU, FRD ↔ FLU) | İleri-geri dönüşüm birim matris vermeli | Kesin |
| **M1:** `robomodel build` | Küre ve elipsoit için tam rapor | Tüm testler yeşil |

### Faz 2. Hidrodinamik: 10–14 saat (**tamam**; kanat O6, REMUS karşılaştırması O5)
| İş | Denenecek | Kabul ölçütü |
|---|---|---|
| Elipsoit eklenmiş kütlesi (Lamb) | Küre ½ρV; L/D = 2, 4, 10 sferoitler | Tablo değerlerinden < %1 (**tamam**, Faz 1'de yapıldı) |
| İnce gövde / şerit teorisi (silindir, Myring) | Uzun sferoitte Lamb ile karşılaştırma | < %5 |
| Kanat ve yüzey eklenmiş kütlesi | REMUS kanatları | Prestero ile < %20 |
| Sönümleme: doğrusal + karesel, ampirik C_d | REMUS `Xuu`, `Yvv`, `Zww`, `Nrr` | Prestero ile < %50, belirsizlik raporda |
| İticiler: T200 eğrisi, dağıtım matrisi B | 6 iticili BlueROV2 düzeni | Kuvvet ve moment dengesi kesin |
| **M2:** tam Fossen parametre seti | Küre ve REMUS | Her değerin belirsizliği var |

### Faz 3. Simülasyon ve doğrulama: 10–13 saat (**simülatör tamam**; REMUS O5)
| İş | Denenecek | Kabul ölçütü |
|---|---|---|
| 6-DOF Fossen simülasyonu (`solve_ivp`) | Durgun suda denge, nötr gövde hareketsiz | Kayma < 1 mm / 60 s |
| Enerji testi | Sönümsüz, kuvvetsiz gövdede kinetik enerji korunuyor | Değişim < %0,1 |
| Son hız | Sabit itkide son hız = √(T / sönüm) | < %1 |
| REMUS 100 doğrulaması | Bizim katsayılar ve Prestero'nunkiler, aynı manevra | Rapor: parametre parametre hata |
| **M3:** REMUS raporu | | A grubu < %5, B grubu < %20 |

### Faz 4. Dışa aktarım: 7–10 saat (**MATLAB tamam**; Gazebo O1)
| İş | Denenecek | Kabul ölçütü |
|---|---|---|
| MATLAB `.m` + doğrusal model (A, B) | MATLAB'da `ode45` sonucu ile Python simülasyonu | Aynı yörünge (< %1) |
| Gazebo SDF (Hydrodynamics eklentisi) | Pencere açmadan Gazebo'da dalış testi | Python simülasyonuyla < %5 |
| Eklenmiş kütle kararlılık uyarısı | `auv_gazebo`'da görülen patlama durumu | Uyarı veriyor |
| **M4 = MVP** | | ~45–55 saat |

### Faz 5. Doğruluk artırma: 12–20 saat
- Capytaine ile ağdan eklenmiş kütle (BEM). Kabul: küre ve sferoitte Lamb'den < %2.
- BlueROV2 (açık kasa). Kabul: von Benzon değerleriyle karşılaştırma raporu.

### Faz 6. Kalibrasyon: 12–18 saat
- Test planı üretici: hangi manevra, kaç saniye, hangi itki
- CSV / rosbag okuma
- Belirsizlik aralığını başlangıç bilgisi (prior) alan parametre oturtma
- Kabul: Gazebo'da "gerçek" diye kaydedilen veriden, bilinen parametreleri < %5 hatayla
  geri bulmak. Ayrı doğrulama kaydında uyum ≥ %90.

### Faz 7. Paylaşım: 8–12 saat
- HTML rapor, doküman, örnekler, paket
- Teknofest takımlarına deneme sürümü

**Sualtı toplamı:** MVP ~45–55 saat, tam sürüm ~85–105 saat.

## 4. Diğer robot kategorileri

Çekirdek (geometri, rijit gövde, aktüatörler, kalibrasyon, dışa aktarım) ortak. Her kategori
sadece **ortam kuvvetlerini** ekler. Sıra, sualtı MVP'si bittikten sonra.

| Kategori | Ortama özel kısım | CAD'den (A/B grubu) | Kalibrasyon gerektiren (C grubu) | Doğrulama gövdesi | Ek süre |
|---|---|---|---|---|---|
| **Tekerlekli rover** | Tekerlek-zemin teması, diferansiyel / Ackermann sürüş | Kütle, eylemsizlik, tekerlek yarıçapı, iz genişliği | Kayma, yuvarlanma direnci, motor sabiti | TurtleBot benzeri basit model, Gazebo | 20–30 s |
| **Su üstü aracı** | Su hattı, kısmi batma, dalga | Su hattı, metasantr yüksekliği | Sönümleme | Kutu tekne (analitik GM) | 15–25 s |
| **Multirotor drone** | Rotor itkisi ve tork, gövde sürüklemesi | Kütle, eylemsizlik, kol uzunluğu | İtki katsayısı, rotor sürüklemesi | Simetrik quadrotor (analitik) | 15–25 s |
| **Robot kol** | Çok gövdeli dinamik, eklemler | Her linkin kütle ve eylemsizliği | Eklem sürtünmesi, dişli boşluğu | 2 linkli sarkaç (analitik) | 25–40 s |
| Sabit kanat | Aerodinamik | Kanat alanı, açıklık | Aerodinamik katsayılar | AVL ile karşılaştırma | Sonra |
| Bacaklı | Temas, çarpma | Linkler | Temas modeli | | Kapsam dışı |

Robot kol için hazır araçlar (Pinocchio, URDF dışa aktarıcılar) güçlü olduğu için o
kategoride hedef, onlarla uyumlu URDF üretmek; yeniden yazmak değil.

## 5. Durum

**Kesin kapsam (K1–K8) yazıldı, 78 test geçiyor:**

| # | Özellik | Kanıt (test) |
|---|---|---|
| K1 | Geometri, rijit gövde, hidrostatik | Hacim < %0,02, eylemsizlik < %0,1 (analitik) |
| K2 | Şekilli parçalar | Kutu ve silindir eylemsizliği formülle birebir |
| K3 | Eklenmiş kütle | Elipsoit: Lamb kesin; mesh olarak verilen sferoitte şerit teorisi Lamb'den < %1 |
| K4 | Coriolis | Hiç iş yapmıyor (νᵀCν = 0), Munk momentinin işareti doğru |
| K5 | Sönümleme | Silindirde çapraz akış formülüyle birebir; değerler tahmini (±%50–100) |
| K6 | İticiler | Dağıtım matrisinde ileri, sapma ve yunuslama işaretleri doğru |
| K7 | Simülatör | Durgun denge, enerji korunumu < %0,1, son hız < %0,1, yuvarlanma periyodu < %1 |
| K8 | MATLAB çıktısı | Dosyalar ve değerler tam hassasiyette; MATLAB'da ode45 ile Python karşılaştırması |

**Bilinen sınırlar (kesin kapsamda):**
- Şerit teorisi kutu gibi kısa gövdelerde zayıf (±%35). Bunun çözümü O2 (BEM).
- Kanat ve uzantılar yok (O6). Torpido araçlarda yunuslama ve sapma sönümü eksik kalır.
- Sönümleme el kitabı tahmini. Doğru değer için O3 (havuz kalibrasyonu) gerekir.
- T200 itki sınırları hafızadan, datasheet'le kontrol edilmeli.

**Sırada:** isteğe bağlı özellikler (O1–O7) konuşulacak.
