#pragma once

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

public:
    explicit Section(const QString & title = "", const int animationDuration = 100, QWidget* parent = 0);

    void setContentLayout(QLayout & contentLayout);
};

