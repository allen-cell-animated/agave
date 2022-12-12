#pragma once

#include <QDialog>

class QSpinBox;
class QTreeWidget;

class LoadDialog : public QDialog
{
  Q_OBJECT

public:
  LoadDialog(std::string path, QWidget* parent = Q_NULLPTR);

private slots:
  void updateScene(int value);

private:
  std::string mPath;
  int mScene;

  QSpinBox* mSceneInput;
  QTreeWidget* mMetadataTree;

  // show multiresolutions

  // select region of interest in zyx
  // select any set of channels
  // start with a single timepoint
};
