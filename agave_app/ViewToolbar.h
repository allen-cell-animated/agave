#pragma once

#include <QWidget>
#include <QPushButton>

class ViewToolbar : public QWidget
{
  Q_OBJECT
public:
  ViewToolbar(QWidget* parent = nullptr);
  virtual ~ViewToolbar();

  QPushButton* topViewButton;
  QPushButton* frontViewButton;
  QPushButton* sideViewButton;

  void positionToolbar();
};
