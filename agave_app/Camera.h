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
  QCamera(QObject* pParent = NULL);
  virtual ~QCamera(void);
  QCamera(const QCamera& Other);
  QCamera& operator=(const QCamera& Other);

  QFilm& GetFilm(void);
  void SetFilm(const QFilm& Film);
  QAperture& GetAperture(void);
  void SetAperture(const QAperture& Aperture);
  QProjection& GetProjection(void);
  void SetProjection(const QProjection& Projection);
  QFocus& GetFocus(void);
  void SetFocus(const QFocus& Focus);

  static QCamera Default(void);

signals:
  void Changed();

private:
  QFilm m_Film;
  QAperture m_Aperture;
  QProjection m_Projection;
  QFocus m_Focus;
};
