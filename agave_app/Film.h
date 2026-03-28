#pragma once

#include <QObject>

class QFilm : public QObject
{
  Q_OBJECT

public:
  QFilm(QObject* pParent = nullptr);
  QFilm(const QFilm& Other);
  QFilm& operator=(const QFilm& Other);

  int GetWidth() const;
  void SetWidth(const int& Width);
  int GetHeight() const;
  void SetHeight(const int& Height);
  float GetExposure() const;
  void SetExposure(const float& Exposure);
  int GetExposureIterations() const;
  void SetExposureIterations(const int& ExposureIterations);
  bool GetNoiseReduction() const;
  void SetNoiseReduction(const bool& NoiseReduction);
  bool IsDirty() const;
  void UnDirty();

signals:
  void Changed(const QFilm& Film);

private:
  int m_Width;
  int m_Height;
  float m_Exposure;
  int m_ExposureIterations;
  bool m_NoiseReduction;
  int m_Dirty;
};
