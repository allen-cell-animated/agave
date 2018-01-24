#include <QPropertyAnimation>

#include "Section.h"

Section::Section(const QString & title, const int animationDuration, bool is_checked, QWidget* parent)
    : QWidget(parent), animationDuration(animationDuration)
{
	checkBox = new QCheckBox(this);
	checkBox->setChecked(is_checked);
    toggleButton = new QToolButton(this);
    headerLine = new QFrame(this);
    toggleAnimation = new QParallelAnimationGroup(this);
    contentArea = new QScrollArea(this);
    mainLayout = new QGridLayout(this);

    toggleButton->setStyleSheet("QToolButton {border: none;}");
    toggleButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    toggleButton->setArrowType(Qt::ArrowType::RightArrow);
    toggleButton->setText(title);
    toggleButton->setCheckable(true);
    toggleButton->setChecked(false);

    headerLine->setFrameShape(QFrame::HLine);
    headerLine->setFrameShadow(QFrame::Sunken);
    headerLine->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);

    contentArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    // start out collapsed
    contentArea->setMaximumHeight(0);
    contentArea->setMinimumHeight(0);

    // let the entire widget grow and shrink with its content
    toggleAnimation->addAnimation(new QPropertyAnimation(this, "minimumHeight"));
    toggleAnimation->addAnimation(new QPropertyAnimation(this, "maximumHeight"));
    toggleAnimation->addAnimation(new QPropertyAnimation(contentArea, "maximumHeight"));

    mainLayout->setVerticalSpacing(0);
    mainLayout->setContentsMargins(0, 0, 0, 0);

    int row = 0;
    mainLayout->addWidget(toggleButton, row, 0, 1, 1, Qt::AlignLeft);
	mainLayout->addWidget(headerLine, row, 2, 1, 1);
	mainLayout->addWidget(checkBox, row++, 3, 1, 1, Qt::AlignRight);
	mainLayout->addWidget(contentArea, row, 0, 1, 4);
    setLayout(mainLayout);

    QObject::connect(toggleButton, &QToolButton::clicked, [this](const bool checked)
    {
        toggleButton->setArrowType(checked ? Qt::ArrowType::DownArrow : Qt::ArrowType::RightArrow);
        toggleAnimation->setDirection(checked ? QAbstractAnimation::Forward : QAbstractAnimation::Backward);
        toggleAnimation->start();
    });

	QObject::connect(checkBox, &QCheckBox::clicked, [this](const bool is_checked)
	{
		emit checked(is_checked);
	});
}

void Section::setContentLayout(QLayout & contentLayout)
{
    delete contentArea->layout();
    contentArea->setLayout(&contentLayout);
    const auto collapsedHeight = sizeHint().height() - contentArea->maximumHeight();
    auto contentHeight = contentLayout.sizeHint().height();

    for (int i = 0; i < toggleAnimation->animationCount() - 1; ++i)
    {
        QPropertyAnimation* SectionAnimation = static_cast<QPropertyAnimation *>(toggleAnimation->animationAt(i));
        SectionAnimation->setDuration(animationDuration);
        SectionAnimation->setStartValue(collapsedHeight);
        SectionAnimation->setEndValue(collapsedHeight + contentHeight);
    }

    QPropertyAnimation* contentAnimation = static_cast<QPropertyAnimation *>(toggleAnimation->animationAt(toggleAnimation->animationCount() - 1));
    contentAnimation->setDuration(animationDuration);
    contentAnimation->setStartValue(0);
    contentAnimation->setEndValue(contentHeight);
}
