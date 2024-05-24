#pragma once

#include <QWidget>

class ViewToolbar : public QWidget
{
  Q_OBJECT
public:
  ViewToolbar(QWidget* parent = nullptr);
  virtual ~ViewToolbar();

  void positionToolbar();
};
