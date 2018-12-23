#include "Focus.h"

QFocus::QFocus(QObject* pParent /*= NULL*/)
  : QObject(pParent)
  , m_FocalDistance(0.75f)
{}

QFocus::QFocus(const QFocus& Other)
{
  *this = Other;
}

QFocus&
QFocus::operator=(const QFocus& Other)
{
  m_FocalDistance = Other.m_FocalDistance;

  emit Changed(*this);

  return *this;
}

int
QFocus::GetType(void) const
{
  return m_Type;
}

void
QFocus::SetType(const int& Type)
{
  m_Type = Type;

  emit Changed(*this);
}

float
QFocus::GetFocalDistance(void) const
{
  return m_FocalDistance;
}

void
QFocus::SetFocalDistance(const float& FocalDistance)
{
  m_FocalDistance = FocalDistance;

  emit Changed(*this);
}

void
QFocus::Reset(void)
{
  m_FocalDistance = 1.0f;

  emit Changed(*this);
}
