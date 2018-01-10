#pragma once

class CScene;

class QTransferFunction : public QObject
{
    Q_OBJECT

public:
    QTransferFunction(QObject* pParent = NULL);
	QTransferFunction(const QTransferFunction& Other);
	QTransferFunction& operator = (const QTransferFunction& Other);			
	
	float						GetDensityScale(void) const;
	void						SetDensityScale(const float& DensityScale);
	int							GetShadingType(void) const;
	void						SetShadingType(const int& ShadingType);
	int							GetRendererType(void) const;
	void						SetRendererType(const int& RendererType);
	float						GetGradientFactor(void) const;
	void						SetGradientFactor(const float& GradientFactor);

	float						GetWindow(void) const;
	void						SetWindow(const float& Window);
	float						GetLevel(void) const;
	void						SetLevel(const float& Level);

	static QTransferFunction	Default(void);

	void setScene(CScene& scene);
	CScene* scene() { return _Scene; }

signals:
	void	Changed(void);
	void	ChangedRenderer(int);

private:
	float		m_DensityScale;
	int			m_ShadingType;
	int m_RendererType;
	float		m_GradientFactor;

	float m_Window;
	float m_Level;

	CScene* _Scene;
};
