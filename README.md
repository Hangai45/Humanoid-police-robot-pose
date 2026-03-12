# 🦾 Police Robot Gesture Recognition (Version 2)

Энэхүү төсөл нь MediaPipe болон Machine Learning (Random Forest) ашиглан хүний гарны хөдөлгөөнийг бодит хугацаанд (real-time) таних систем юм.

## ✨ Гол онцлогууд
- **18 Features (Spatial Awareness)**: 14 үений өнцөг болон чиглэл мэдрэх 4 босоо тэнхлэгийн үзүүлэлттэй.
- **100% Accuracy**: Шинэчилсэн алгоритм нь гарны байрлалыг орон зайд маш нарийн (дээшээ, доошоо, хажуу тийш) ялгадаг.
- **Real-time Performance**: Камерын дүрсийг хурдан боловсруулахын тулд frame skipping болон resolution optimization ашигласан.
- **Automated Workflow**: Дата цуглуулах, нэтгэх, сургах үйл явцыг бүрэн автоматжуулсан.

## 📂 Төслийн бүтэц
- [version_2/dataset.py](cci:7://file:///c:/Users/DELL.DESKTOP-PTQ10MO/Desktop/WEB/police%20robot/version_2/dataset.py:0:0-0:0): Шинэ дата цуглуулагч (18 features).
- [version_2/combine_data.py](cci:7://file:///c:/Users/DELL.DESKTOP-PTQ10MO/Desktop/WEB/police%20robot/version_2/combine_data.py:0:0-0:0): Олон датаг нэгтгэн `merged_features.csv` үүсгэгч.
- [version_2/tanilt.py](cci:7://file:///c:/Users/DELL.DESKTOP-PTQ10MO/Desktop/WEB/police%20robot/version_2/tanilt.py:0:0-0:0): AI сургалт болон бодит хугацаанд танигч.
- [walkthrough_guide.md](cci:7://file:///c:/Users/DELL.DESKTOP-PTQ10MO/Desktop/WEB/police%20robot/walkthrough_guide.md:0:0-0:0): Хэрэглэх заавар.

## 🚀 Технологийн сангууд
- Python 3.11
- MediaPipe (Pose & Hands)
- Scikit-learn (Random Forest)
- OpenCV
- Pandas, NumPy
