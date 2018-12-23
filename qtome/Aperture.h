#pragma once

#include <QObject>

class QAperture : public QObject
{
  Q_OBJECT

public:
  QAperture(QObject* pParent = NULL);
  QAperture(const QAperture& Other);
  QAperture& operator=(const QAperture& Other);

  float GetSize(void) const;
  void SetSize(const float& Size);
  void Reset(void);

signals:
  void Changed(const QAperture& Aperture);

private:
  float m_Size;
};
