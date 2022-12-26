#pragma once

#include "renderlib/VolumeDimensions.h"

#include <QDialog>

class QLabel;
class QSpinBox;
class QTreeWidget;

class RangeWidget;

class LoadDialog : public QDialog
{
  Q_OBJECT

public:
  LoadDialog(std::string path, const std::vector<MultiscaleDims>& dims, QWidget* parent = Q_NULLPTR);

private slots:
  void updateScene(int value);
  void onItemSelectionChanged();

private:
  std::string mPath;
  int mScene;
  std::vector<MultiscaleDims> mDims;

  QSpinBox* mSceneInput;
  QTreeWidget* mMetadataTree;
  QLabel* mMemoryEstimateLabel;
  RangeWidget* m_roiX;
  RangeWidget* m_roiY;
  RangeWidget* m_roiZ;
  // show multiresolutions

  // select region of interest in zyx
  // select any set of channels
  // start with a single timepoint

  void updateMemoryEstimate();
};
