module.exports = {
  extends: [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "prettier",
  ],
  env: {
    browser: true,
    es6: true,
    mocha: true,
  },
  plugins: ["@typescript-eslint"],
  rules: {
    "@typescript-eslint/ban-types": ["warn"],
    "@typescript-eslint/naming-convention": [
      "warn",
      {
        selector: "default",
        format: ["camelCase", "PascalCase"],
      },
      {
        selector: "variable",
        format: ["camelCase", "UPPER_CASE"],
      },
      {
        selector: "property",
        format: ["camelCase", "UPPER_CASE"],
      },
      {
        selector: "typeLike",
        format: ["PascalCase"],
      },
      {
        selector: "enumMember",
        format: ["UPPER_CASE"],
      },
      {
        selector: "parameter",
        format: ["camelCase"],
        leadingUnderscore: "allow",
      },
    ],
    "@typescript-eslint/indent": ["off"],
    "@typescript-eslint/no-empty-function": ["warn"],
    "@typescript-eslint/no-inferrable-types": ["warn"],
    "@typescript-eslint/no-this-alias": ["warn"],
    "prefer-const": ["warn"],
    "prefer-spread": ["warn"],
    "no-var": ["warn"],
    // note you must disable the base rule as it can report incorrect errors
    "no-unused-vars": "off",
    "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
  },
};
