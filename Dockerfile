FROM ruby:3.3

WORKDIR /site

RUN gem install bundler -v 2.5.22

COPY Gemfile Gemfile.lock ./
RUN bundle _2.5.22_ config set path /bundle \
  && bundle _2.5.22_ lock --add-platform ruby x86_64-linux \
  && bundle _2.5.22_ install

EXPOSE 4000 35729

CMD ["bundle", "_2.5.22_", "exec", "jekyll", "serve", "--host", "0.0.0.0", "--livereload", "--force_polling"]
