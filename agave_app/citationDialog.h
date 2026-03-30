#pragma once

#include <QDialog>

class CitationDialog : public QDialog
{
public:
  CitationDialog();
  ~CitationDialog() override;

  QSize sizeHint() const override { return QSize(500, 100); }
};
