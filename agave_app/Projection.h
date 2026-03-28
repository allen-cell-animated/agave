#pragma once

#include <QObject>

class QProjection : public QObject
{
  Q_OBJECT

public:
  QProjection(QObject* pParent = nullptr);
  QProjection(const QProjection& Other);
  QProjection& operator=(const QProjection& Other);

  float GetFieldOfView() const;
  void SetFieldOfView(const float& FieldOfView);
  void Reset();

signals:
  void Changed(const QProjection& Projection);

private:
  float m_FieldOfView;
};
