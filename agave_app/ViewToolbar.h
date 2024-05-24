#pragma once

#include <QPushButton>
#include <QWidget>

class ViewToolbar : public QWidget
{
  Q_OBJECT
public:
  ViewToolbar(QWidget* parent = nullptr);
  virtual ~ViewToolbar();

  QPushButton* frameViewButton;
  QPushButton* orthoViewButton;
  QPushButton* topViewButton;
  QPushButton* bottomViewButton;
  QPushButton* frontViewButton;
  QPushButton* backViewButton;
  QPushButton* leftViewButton;
  QPushButton* rightViewButton;

  void positionToolbar();
};
