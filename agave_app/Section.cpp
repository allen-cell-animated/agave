#include <QPropertyAnimation>

#include "Section.h"

#include <QLabel>

Section::Section(const QString& title, const int animationDuration, const CheckBoxInfo* checkBoxInfo, QWidget* parent)
  : QWidget(parent)
  , m_animationDuration(animationDuration)
  , m_checkBox(nullptr)
{
  m_toggleButton = new QToolButton(this);
  m_headerLine = new QFrame(this);
  m_toggleAnimation = new QParallelAnimationGroup(this);
  m_contentArea = new QScrollArea(this);
  m_mainLayout = new QGridLayout(this);

  // get standard QLabel font size
  QFont f = QLabel("A").font();
  int px = f.pointSize();

  m_toggleButton->setStyleSheet(QString("QToolButton {border: none; font-size: ") + QString::number(px) +
                                QString("pt;}"));
  m_toggleButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
  m_toggleButton->setArrowType(Qt::ArrowType::RightArrow);
  m_toggleButton->setText(title);
  m_toggleButton->setCheckable(true);
  m_toggleButton->setChecked(false);

  m_headerLine->setFrameShape(QFrame::HLine);
  m_headerLine->setFrameShadow(QFrame::Sunken);
  m_headerLine->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);

  m_contentArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

  // start out collapsed
  m_contentArea->setMaximumHeight(0);
  m_contentArea->setMinimumHeight(0);

  // let the entire widget grow and shrink with its content
  m_toggleAnimation->addAnimation(new QPropertyAnimation(this, "minimumHeight"));
  m_toggleAnimation->addAnimation(new QPropertyAnimation(this, "maximumHeight"));
  m_toggleAnimation->addAnimation(new QPropertyAnimation(m_contentArea, "maximumHeight"));

  m_mainLayout->setVerticalSpacing(0);
  m_mainLayout->setContentsMargins(0, 0, 0, 0);

  int row = 0;
  m_mainLayout->addWidget(m_toggleButton, row, 0, 1, 1, Qt::AlignLeft);
  m_mainLayout->addWidget(m_headerLine, row, 2, 1, 1);
  bool use_checkbox = checkBoxInfo != nullptr;
  if (use_checkbox) {
    m_checkBox = new QCheckBox(this);
    m_checkBox->setChecked(checkBoxInfo->is_checked);
    m_checkBox->setToolTip(QString::fromStdString(checkBoxInfo->toolTip));
    m_checkBox->setStatusTip(QString::fromStdString(checkBoxInfo->statusTip));
    m_mainLayout->addWidget(m_checkBox, row, 3, 1, 1, Qt::AlignRight);
    QObject::connect(m_checkBox, &QCheckBox::clicked, [this](const bool is_checked) { emit checked(is_checked); });
  }
  row++;
  m_mainLayout->addWidget(m_contentArea, row, 0, 1, 4);
  setLayout(m_mainLayout);

  QObject::connect(m_toggleButton, &QToolButton::clicked, [this](const bool checked) {
    m_toggleButton->setArrowType(checked ? Qt::ArrowType::DownArrow : Qt::ArrowType::RightArrow);
    m_toggleAnimation->setDirection(checked ? QAbstractAnimation::Forward : QAbstractAnimation::Backward);
    m_toggleAnimation->start();
  });
  QObject::connect(m_toggleAnimation, &QParallelAnimationGroup::finished, [this]() {
    if (m_toggleAnimation->direction() == QAbstractAnimation::Backward) {
      m_contentArea->setVisible(false);
      emit collapsed();
    } else {
      m_contentArea->setVisible(true);
      emit expanded();
    }
  });
}

void
Section::setTitle(const QString& title)
{
  m_toggleButton->setText(title);
}

void
Section::setContentLayout(QLayout& contentLayout)
{
  delete m_contentArea->layout();
  m_contentArea->setLayout(&contentLayout);
  const auto collapsedHeight = sizeHint().height() - m_contentArea->maximumHeight();
  auto contentHeight = contentLayout.sizeHint().height();

  for (int i = 0; i < m_toggleAnimation->animationCount() - 1; ++i) {
    QPropertyAnimation* SectionAnimation = static_cast<QPropertyAnimation*>(m_toggleAnimation->animationAt(i));
    SectionAnimation->setDuration(m_animationDuration);
    SectionAnimation->setStartValue(collapsedHeight);
    SectionAnimation->setEndValue(collapsedHeight + contentHeight);
  }

  QPropertyAnimation* contentAnimation =
    static_cast<QPropertyAnimation*>(m_toggleAnimation->animationAt(m_toggleAnimation->animationCount() - 1));
  contentAnimation->setDuration(m_animationDuration);
  contentAnimation->setStartValue(0);
  contentAnimation->setEndValue(contentHeight);
}

bool
Section::isChecked() const
{
  if (m_checkBox) {
    return m_checkBox->isChecked();
  }
  return false;
}

void
Section::setChecked(bool checked)
{
  if (m_checkBox) {
    m_checkBox->setChecked(checked);
  }
}