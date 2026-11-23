# Doğrulama: bilinen robotlar (TurtleBot3, Franka Panda)

Betikler: `validation/turtlebot3.py`, `validation/panda.py`. Veriler `data/` altında (git'e girmez;
`~/veri/veri/kaynaklar/` klasörüne bağlantı).

## TurtleBot3 Burger (kara robotu)
Kaynak: ROBOTIS e-manual (azami 0,22 m/s, 2,84 rad/s) ve Dynamixel XL430 tablosu (1,4 N·m,
57 rpm @11,1 V; 61 rpm @12 V). Model: ROBOTIS'in gövde mesh'i, URDF'teki tekerlek konumları.

| | robomodel | ROBOTIS | Fark |
|---|---|---|---|
| Son hız, 11,1 V / 12 V | 0,197 / 0,211 m/s | 0,22 m/s | −%11 / −%4 |
| Yerinde dönüş, 12 V | 2,63 rad/s | 2,84 rad/s | −%7 |
| Gövde ataleti Izz / Ixx | eşit yoğunluk | üreticinin CAD'i | %0 / +%31 |

## Franka Panda (robot kol)
Referans: Gaz et al., IEEE RA-L 2019. Tanımlanmış parametreler ek materyalin Tablo VIII–IX'undan
alındı; model gerçek robotta ölçülen torkları tekrarlıyor. robomodel: Franka'nın çarpışma
mesh'leri ve yalnızca link kütleleri; kütle merkezi ve atalet eşit yoğunlukla CAD'den. Karşılaştırma
için üreticinin kendi `inertials.yaml` değerleri de var. 200 rastgele poz.

| Yerçekimi torku RMS hatası | j2 | j3 | j4 | j5 | j6 |
|---|---|---|---|---|---|
| üretici değerleri | %10 | %14 | %16 | %35 | %35 |
| robomodel (yalnızca CAD + kütleler) | %14 | %18 | %21 | %44 | %45 |

Kütle matrisi M(q), ortanca bağıl hata: üretici %18, robomodel %22.

j1 (dikey eksen) yerçekimi torku taşımaz. j7'nin torku çok küçük olduğu için yüzdesi anlamsız.

## Sonuç
- **Hız ve dönüş:** motor tablosundan %4–11 içinde.
- **Kol:** yalnızca CAD ve tartılmış kütlelerle, üreticinin kendi değerlerine yakın (4–9 puan
  geride).
- **Uç eklemler** (j5–j7) ile eşit yoğunluklu atalet zayıf: motor ve dişli gibi yoğun iç parçalar
  ve rotor ataleti CAD'de yok. STEP montajı (parça parça kütle) ve rotor ataleti bunu iyileştirir.
