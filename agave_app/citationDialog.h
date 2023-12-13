#pragma once

#include <QDialog>

class CitationDialog : public QDialog
{
public:
  CitationDialog();
  virtual ~CitationDialog();

  QSize sizeHint() const override { return QSize(500, 100); }
};
