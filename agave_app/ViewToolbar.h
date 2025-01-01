#pragma once

#include "Controls.h"

#include <QPushButton>
#include <QToolButton>
#include <QWidget>

class CCamera;

class ViewToolbar : public QWidget
{
  Q_OBJECT
public:
  ViewToolbar(QWidget* parent = nullptr);
  virtual ~ViewToolbar();
  void initFromCamera(const CCamera& camera);

  QPushButton* axisViewButton;

  QPushButton* homeButton;
  QPushButton* frameViewButton;
  DualIconButton* orthoViewButton;
  QPushButton* topViewButton;
  QPushButton* bottomViewButton;
  QPushButton* frontViewButton;
  QPushButton* backViewButton;
  QPushButton* leftViewButton;
  QPushButton* rightViewButton;
  QPushButton* axisHelperButton;
  QPushButton* rotateButton;
  QPushButton* translateButton;
};
