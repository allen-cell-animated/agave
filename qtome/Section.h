#pragma once

#include <QCheckBox>
#include <QFrame>
#include <QGridLayout>
#include <QParallelAnimationGroup>
#include <QScrollArea>
#include <QToolButton>
#include <QWidget>

class Section : public QWidget {
    Q_OBJECT
private:

    QGridLayout* mainLayout;
    QToolButton* toggleButton;
    QFrame* headerLine;
    QParallelAnimationGroup* toggleAnimation;
    QScrollArea* contentArea;
    int animationDuration;

	QCheckBox* checkBox;

public:
    explicit Section(const QString & title = "", const int animationDuration = 100, bool is_checked = true, QWidget* parent = 0);

    void setContentLayout(QLayout & contentLayout);

signals:
	void checked(bool checked);

};

