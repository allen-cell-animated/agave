#include "Aperture.h"

QAperture::QAperture(QObject* pParent /*= NULL*/)
  : QObject(pParent)
  , m_Size(0.0f)
{}

QAperture::QAperture(const QAperture& Other)
{
  *this = Other;
}

QAperture&
QAperture::operator=(const QAperture& Other)
{
  m_Size = Other.m_Size;

  emit Changed(*this);

  return *this;
}

float
QAperture::GetSize(void) const
{
  return m_Size;
}

void
QAperture::SetSize(const float& Size)
{
  m_Size = Size;

  emit Changed(*this);
}

void
QAperture::Reset(void)
{
  m_Size = 0.0f;

  emit Changed(*this);
}
