#pragma once

#include <QObject>

class QAperture : public QObject
{
  Q_OBJECT

public:
  QAperture(QObject* pParent = NULL);
  QAperture(const QAperture& Other);
  QAperture& operator=(const QAperture& Other);

  float GetSize() const;
  void SetSize(const float& Size);
  void Reset();

signals:
  void Changed(const QAperture& Aperture);

private:
  float m_Size;
};
