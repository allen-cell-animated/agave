const path = require("path");
const webpack = require("webpack");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const { CleanWebpackPlugin } = require("clean-webpack-plugin");

module.exports = {
  entry: ["./public/index.ts"],
  output: {
    path: path.resolve(__dirname, "agave-ui"),
    filename: "agave-ui.bundle.js",
  },
  devtool: "source-map",
  devServer: {
    open: ["/"],
    port: 9020,
    static: [
      {
        staticOptions: {
          dotfiles: "allow",
        },
      },
    ],
  },
  performance: {
    hints: false,
  },
  mode: "development",
  plugins: [
    new CleanWebpackPlugin(),
    new webpack.DefinePlugin({
      APP_VERSION: JSON.stringify(require("./package.json").version),
    }),
    new HtmlWebpackPlugin({
      template: "./public/index.html",
    }),
  ],
  resolve: {
    extensions: [".js", ".ts"],
  },
  module: {
    rules: [
      {
        test: /\.(js|ts)$/,
        exclude: /node_modules/,
        use: "babel-loader",
      },
    ],
  },
};
