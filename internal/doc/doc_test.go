package doc

import "testing"

func TestRoleOf(t *testing.T) {
	cases := map[string]Role{
		"title": RoleBody, "section_header": RoleBody, "text": RoleBody, "list_item": RoleBody, "formula": RoleBody, "code": RoleBody,
		"picture": RoleInserted, "table": RoleInserted, "caption": RoleInserted, "footnote": RoleInserted,
		"page_header": RoleDropped, "page_footer": RoleDropped,
	}
	for kind, want := range cases {
		if got := RoleOf(kind); got != want {
			t.Errorf("%s: got %s, want %s", kind, got, want)
		}
	}
}
