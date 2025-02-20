#pragma once

#include <QCheckBox>
#include <QFrame>
#include <QGridLayout>
#include <QParallelAnimationGroup>
#include <QScrollArea>
#include <QToolButton>
#include <QWidget>

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

public:
  struct CheckBoxInfo
  {
    bool is_checked;
    std::string toolTip;
    std::string statusTip;
  };

  explicit Section(const QString& title = "",
                   const int animationDuration = 100,
                   const CheckBoxInfo* checkBoxInfo = nullptr,
                   QWidget* parent = 0);

  void setContentLayout(QLayout& contentLayout);
  void setTitle(const QString& title);

  bool isChecked() const;
  void setChecked(bool checked);

signals:
  void checked(bool checked);
  void collapsed();
  void expanded();
};
