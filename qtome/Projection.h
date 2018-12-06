#pragma once

#include <QObject>

class QProjection : public QObject
{
  Q_OBJECT

public:
  QProjection(QObject* pParent = NULL);
  QProjection(const QProjection& Other);
  QProjection& operator=(const QProjection& Other);

  float GetFieldOfView(void) const;
  void SetFieldOfView(const float& FieldOfView);
  void Reset(void);

signals:
  void Changed(const QProjection& Projection);

private:
  float m_FieldOfView;
};
