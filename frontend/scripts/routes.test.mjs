import { test } from "node:test";
import assert from "node:assert/strict";
import { pageFromHash, PAGE_PATHS } from "../src/routes.ts";
test("All three navigation routes resolve to their intended page", () => {
  for (const page of [
    "overview",
    "classifier",
    "about",
  ])
    assert.equal(pageFromHash("#/" + page), page);
});
test("Classifier slash URLs never fall back to Overview", () => {
  assert.equal(pageFromHash("#/classifier"), "classifier");
  assert.equal(pageFromHash("#/classifier/"), "classifier");
});
test("Existing bookmarks retain their destination", () => {
  assert.equal(pageFromHash("#classifier"), "classifier");
});
test("Every navigation destination has a canonical slash URL", () => {
  assert.deepEqual(Object.values(PAGE_PATHS), [
    "#/overview",
    "#/classifier",
    "#/about",
  ]);
});
test("Root, malformed, and prototype names resolve safely", () => {
  for (const hash of ["", "#/missing", "#/__proto__", "#/constructor"])
    assert.equal(pageFromHash(hash), "overview");
});

test("Retired evaluation bookmarks redirect to About", () => {
  for (const hash of ["#evaluation", "#/evaluation", "#/evaluation/"])
    assert.equal(pageFromHash(hash), "about");
});

test("Removed dataset routes fall back to Overview", () => {
  for (const hash of ["#datasets", "#/datasets", "#/datasets/", "#explorer", "#/explorer"])
    assert.equal(pageFromHash(hash), "overview");
});
