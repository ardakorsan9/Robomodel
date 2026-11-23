# Doğrulama: su üstü araçları (tekne, USV)

Betik: `uv run python validation/surface_wigley.py` · Testler: `tests/test_surface.py`

## Yöntem

1. **Yüzme dengesi.** Ağırlık, taşınan suyun ağırlığına eşit olmalı ve ağırlık merkezi (CG)
   kaldırma merkezinin (CB) tam üstünde durmalı. Su altındaki hacim ve onun merkezi kapalı
   CAD gövdesinden **tam** hesaplanıyor: diverjans teoremiyle yalnızca gövde yüzeyinin ıslak
   kısmı kullanılıyor, su hattı kesiti hiç gerekmiyor. Su çekimi, trim ve yatma açısı Newton
   yöntemiyle bulunuyor.
2. **Geri getirme kuvvetleri.** `G` matrisi (dikey, yatma, trim yayları) denge noktasında
   sayısal türevle hesaplanıyor; buradan su hattı alanı, GM_T ve GM_L çıkıyor. Büyük yatma
   açıları için GZ eğrisi ±90° aralığında, her açıda dikey konum ve trim yeniden dengelenerek
   hesaplanıyor. Simülasyon yatmada bu eğriyi, diğer eksenlerde `G`'yi kullanıyor.
3. **Ek kütle ve sönüm.** Islak gövde ile onun su yüzeyindeki aynası birlikte tek bir cisim
   ("çift gövde") olarak sonsuz akışkanda hesaplanıyor (şerit teorisi / BEM / elipsoit), sonra
   ikiye bölünüyor. Bu, manevra için alışılmış limitleri doğrudan veriyor: ileri, yan ve
   dönüşte düşük frekans (yüzey duvar gibi), dikey, yatma ve trimde yüksek frekans (yüzeyde
   basınç sıfır).
4. **Dikey, yatma ve trimde doğrusal sönüm.** Dalga yayılımı ve yalpa omurgası etkileri
   CAD'den çıkmıyor. Kritik sönümün tipik bir payı alınıyor: dikey ve trim 0,2, yatma 0,05
   (varsayım, geniş aralıkla).

## Sonuçlar

### Kutu duba 10 × 4 × 2 m, su çekimi 1 m, KG 1 m (el formülleri)

| Büyüklük | robomodel | Analitik | Fark |
|---|---|---|---|
| Hacim | 40 m³ | 40 m³ | 0,000 % |
| Su hattı alanı | 40 m² | 40 m² | 0,000 % |
| GM_T = KB + B²/(12T) − KG | 0,83333 m | 0,83333 m | 0,000 % |
| GM_L = KB + L²/(12T) − KG | 7,8333 m | 7,8333 m | 0,000 % |
| GZ 10° / 20° / 25° (dik duvar formülü) | 0,14831 / 0,31522 / 0,41345 m | aynı | 0,00 % |
| CG 0,3 m önde: tan(trim) = −x_g / GM_L | −2,19° | −2,19° | < 2 % (küçük açı yaklaşımı) |

Dik duvar formülü güverte kenarı suya girene (26,6°) kadar geçerli; robomodel bu açıdan sonra
da gerçek geometriyi kullanmaya devam ediyor.

### Wigley gövdesi L = 4 m, B = 0,4 m, T = 0,25 m (kapalı formüller)

| Büyüklük | robomodel | Analitik | Fark |
|---|---|---|---|
| Blok katsayısı C_B | 0,44439 | 4/9 | −0,012 % |
| Su hattı katsayısı C_WP | 0,66664 | 2/3 | −0,005 % |
| KB | 0,15625 m | 5T/8 | −0,001 % |
| GM_T = KB + 3B²/(35T) − KG | 0,011107 m | 0,011107 m | −0,006 % |

Farklar yalnızca gövdenin üçgenlerle temsil edilmesinden kaynaklanıyor.

### Yarısı batık küre: ek kütle

Ayna yöntemiyle tam küre elde ediliyor, bu yüzden her ötelemede ek kütle, taşınan suyun
kütlesinin 0,5 katı olmalı. robomodel'de ileri, yan ve dikey eksenlerin üçü de %6 içinde
çıkıyor (test).

### Manevra katsayıları: Clarke, Gedling & Hine (1983) regresyonuyla karşılaştırma

Clarke formülleri birçok gemi modelinin deney sonuçlarından türetilmiş bir regresyon. Burada
Fossen'in *Handbook of Marine Craft Hydrodynamics and Motion Control* (2. baskı, 2021) kitabında
verildiği haliyle kullanıldı; **formüller kaynaktan elle yazıldı, kullanmadan önce kitaptan
kontrol edin**. Gerçek ölçüm değil, ölçümlere uydurulmuş bir eğri. Wigley gövdesi bu
regresyonun kapsadığı ticari gemi formlarından daha ince (B/T = 1,6).

Boyutsuz değerler, hepsi −X' biçiminde, u = 1 m/s:

| Katsayı | robomodel | Clarke | Fark | Yorum |
|---|---|---|---|---|
| Y_v̇ (yan ek kütle) | 0,01176 | 0,01304 | −10 % | iyi |
| N_ṙ (dönüş ek ataleti) | 0,000908 | 0,000766 | +18 % | kabul edilebilir |
| Y_v (yan kaymada kuvvet) | 0,01573 | 0,01576 | −0 % | çok iyi (Hoerner gövde kaldırması) |
| N_v (yan kaymada moment, Munk dahil) | 0,01031 | 0,00798 | +29 % | kaba |
| Y_r (dönüşte yan kuvvet) | 0,00012 | −0,0050 | yok | **eksik**: dönüşün gövde boyunca yarattığı kaldırma modellenmiyor |
| N_r (dönüş sönümü) | 0,00172 | 0,00315 | −45 % | **düşük**: tekne gerçekte olduğundan hızlı döner |

## Sonuç: neye güvenilir?

- **Yüzme dengesi, su çekimi, trim, GM, GZ eğrisi:** CAD kapalı ve CG doğruysa tam. Hata
  kaynağı CAD değil, **CG yüksekliği**: GM doğrudan buna bağlı. Ağır parçaları listeleyin veya
  bir eğim testi yapın (`measured.r_g`). Ağır parça listesi yoksa gövde kütlesi yüzeye
  yayılmış kabul ediliyor (`hull_mass: shell`, ince cidarlı tekne).
- **Ek kütle:** yan ve dikey eksenlerde %10 civarında.
- **Yan kayma kuvveti (Y_v):** iyi.
- **Dönüş türevleri (Y_r, N_r):** zayıf. Dönüş hızı tahmini, REMUS'taki gibi olması
  gerekenden yüksek. Bir dönüş testiyle `robomodel calibrate` yapın veya low/high aralığını
  kullanın.
- **Direnç:** sürtünme ağırlıklı (Hoerner); Froude 0,3'ün üstündeki dalga direnci ve kızaklama
  modellenmiyor, hızlı teknelerde son hız yüksek çıkar.
- **Dalga kuvvetleri ve frekansa bağlı katsayılar** modelde yok: sakin su manevra modeli.

## Henüz yapılmayanlar

- Y_r, N_r için gövde boyunca dağılmış kaldırma (veya ince gemi formları için Clarke
  regresyonunu seçenek olarak sunmak).
- Gazebo çıktısı: gz-sim'in su hattı kuvvetleri için hazır bir eklentisi yok; şimdilik MATLAB
  kullanılıyor.
- Gerçek bir USV'nin ölçümüyle karşılaştırma (veri gerekiyor).
