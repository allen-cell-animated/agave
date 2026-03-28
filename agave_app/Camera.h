#pragma once

#include "Aperture.h"
#include "Film.h"
#include "Focus.h"
#include "Projection.h"

class CCamera;
class CScene;

class QCamera : public QObject
{
  Q_OBJECT

public:
  QCamera(QObject* pParent = nullptr);
  ~QCamera() override;
  QCamera(const QCamera& Other);
  QCamera& operator=(const QCamera& Other);

  QFilm& GetFilm();
  void SetFilm(const QFilm& Film);
  QAperture& GetAperture();
  void SetAperture(const QAperture& Aperture);
  QProjection& GetProjection();
  void SetProjection(const QProjection& Projection);
  QFocus& GetFocus();
  void SetFocus(const QFocus& Focus);

  static QCamera Default();

signals:
  void Changed();

private:
  QFilm m_Film;
  QAperture m_Aperture;
  QProjection m_Projection;
  QFocus m_Focus;
};
