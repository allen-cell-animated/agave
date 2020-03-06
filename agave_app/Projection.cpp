#include "Projection.h"

QProjection::QProjection(QObject* pParent /*= NULL*/)
  : QObject(pParent)
  , m_FieldOfView(55.0f)
{}

QProjection::QProjection(const QProjection& Other)
{
  *this = Other;
}

QProjection&
QProjection::operator=(const QProjection& Other)
{
  m_FieldOfView = Other.m_FieldOfView;

  emit Changed(*this);

  return *this;
}

float
QProjection::GetFieldOfView(void) const
{
  return m_FieldOfView;
}

void
QProjection::SetFieldOfView(const float& FieldOfView)
{
  m_FieldOfView = FieldOfView;

  emit Changed(*this);
}

void
QProjection::Reset(void)
{
  m_FieldOfView = 35.0f;

  emit Changed(*this);
}
