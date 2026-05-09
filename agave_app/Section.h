#pragma once

#include <QCheckBox>
#include <QColor>
#include <QFrame>
#include <QGradient>
#include <QGridLayout>
#include <QParallelAnimationGroup>
#include <QScrollArea>
#include <QToolButton>
#include <QWidget>

class QColorPushButton;

class Section : public QWidget
{
  Q_OBJECT
private:
  QGridLayout* m_mainLayout;
  QToolButton* m_toggleButton;
  QFrame* m_headerLine;
  QParallelAnimationGroup* m_toggleAnimation;
  QScrollArea* m_contentArea;
  int m_animationDuration;

  QCheckBox* m_checkBox;
  QColorPushButton* m_colorButton;

public:
  struct CheckBoxInfo
  {
    bool is_checked;
    std::string toolTip;
    std::string statusTip;
  };
  struct ColorBoxInfo
  {
    QColor color;
    std::string toolTip;
    std::string statusTip;
  };

  explicit Section(const QString& title = "",
                   const int animationDuration = 100,
                   const CheckBoxInfo* checkBoxInfo = nullptr,
                   const ColorBoxInfo* colorBoxInfo = nullptr,
                   QWidget* parent = nullptr);

  void setContentLayout(QLayout& contentLayout);
  void setTitle(const QString& title);

  bool isChecked() const;
  void setChecked(bool checked);

  QColor getColor() const;
  void setColor(const QColor& color);

  // Set an optional colormap gradient drawn behind the section's color
  // swatch. Pass empty stops to render a solid color swatch.
  void setColormapStops(const QGradientStops& stops);

signals:
  void checked(bool checked);
  void collapsed();
  void expanded();
  void colorChanged(const QColor& color);
};
