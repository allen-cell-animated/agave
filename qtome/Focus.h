#pragma once

#include <QObject>

class QFocus : public QObject
{
  Q_OBJECT

public:
  QFocus(QObject* pParent = NULL);
  QFocus(const QFocus& Other);
  QFocus& operator=(const QFocus& Other);

  int GetType(void) const;
  void SetType(const int& Type);
  float GetFocalDistance(void) const;
  void SetFocalDistance(const float& FocalDistance);
  void Reset(void);

signals:
  void Changed(const QFocus& Focus);

private:
  int m_Type;
  float m_FocalDistance;
};
