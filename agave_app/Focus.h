#pragma once

#include <QObject>

class QFocus : public QObject
{
  Q_OBJECT

public:
  QFocus(QObject* pParent = NULL);
  QFocus(const QFocus& Other);
  QFocus& operator=(const QFocus& Other);

  int GetType() const;
  void SetType(const int& Type);
  float GetFocalDistance() const;
  void SetFocalDistance(const float& FocalDistance);
  void Reset();

signals:
  void Changed(const QFocus& Focus);

private:
  int m_Type;
  float m_FocalDistance;
};
