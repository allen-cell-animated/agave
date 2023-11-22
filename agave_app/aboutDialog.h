#pragma once

#include <QDialog>

class AboutDialog : public QDialog
{
public:
  AboutDialog();
  virtual ~AboutDialog();

  QSize sizeHint() const override { return QSize(500, 300); }
};
