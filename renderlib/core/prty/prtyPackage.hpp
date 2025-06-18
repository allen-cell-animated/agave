/*****************************************************************************
**  prtyPackage.hpp
**
**		package for property classes
**
**
**
**
\****************************************************************************/

#ifdef PRTY_PACKAGE_HPP
#error prtyPackage.hpp multiply included
#endif
#define PRTY_PACKAGE_HPP

//============================================================================
//============================================================================
class prtyPackage
{
public:
  //------------------------------------------------------------------------
  //	Init must be called before you use this package.  A good place to
  //	do this is in your main function, with your other package initializers.
  //------------------------------------------------------------------------
  static void Init();

  //------------------------------------------------------------------------
  //	CleanUp should be called after you are done with this package.
  //	A good place to do this is in your main function, after you are done
  //	with other deinitialization and cleanup tasks.
  //------------------------------------------------------------------------
  static void CleanUp();
};