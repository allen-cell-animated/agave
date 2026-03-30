#pragma once

#include <QDialog>

class AboutDialog : public QDialog
{
public:
  AboutDialog();
  ~AboutDialog() override;

  QSize sizeHint() const override { return QSize(500, 100); }
};
