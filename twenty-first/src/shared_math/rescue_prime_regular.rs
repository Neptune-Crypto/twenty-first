use itertools::Itertools;
use num_traits::One;
use serde::{Deserialize, Serialize};

use crate::shared_math::b_field_element::{BFieldElement, BFIELD_ONE, BFIELD_ZERO};
use crate::shared_math::traits::FiniteField;
use crate::util_types::algebraic_hasher::AlgebraicHasher;

use super::rescue_prime_digest::{Digest, DIGEST_LENGTH};

pub const STATE_SIZE: usize = 16;
pub const CAPACITY: usize = 6;
pub const RATE: usize = 10;
pub const NUM_ROUNDS: usize = 8;

pub const ALPHA: u64 = 7;
pub const ALPHA_INV: u64 = 10540996611094048183;

pub const MDS: [BFieldElement; STATE_SIZE * STATE_SIZE] = [
    BFieldElement::new(5910257123858819639),
    BFieldElement::new(3449115226714951713),
    BFieldElement::new(16770055338049327985),
    BFieldElement::new(610399731775780810),
    BFieldElement::new(7363016345531076300),
    BFieldElement::new(16174724756564259629),
    BFieldElement::new(8736587794472183152),
    BFieldElement::new(12699016954477470956),
    BFieldElement::new(13948112026909862966),
    BFieldElement::new(18015813124076612987),
    BFieldElement::new(9568929147539067610),
    BFieldElement::new(14859461777592116402),
    BFieldElement::new(18169364738825153183),
    BFieldElement::new(18221568702798258352),
    BFieldElement::new(1524268296724555606),
    BFieldElement::new(5538821761600),
    BFieldElement::new(1649528676200182784),
    BFieldElement::new(336497118937017052),
    BFieldElement::new(15805000027048028625),
    BFieldElement::new(15709375513998678646),
    BFieldElement::new(14837031240173858084),
    BFieldElement::new(11366298206428370494),
    BFieldElement::new(15698532768527519720),
    BFieldElement::new(5911577595727321095),
    BFieldElement::new(16676030327621016157),
    BFieldElement::new(16537624251746851423),
    BFieldElement::new(13325141695736654367),
    BFieldElement::new(9337952653454313447),
    BFieldElement::new(9090375522091353302),
    BFieldElement::new(5605636660979522224),
    BFieldElement::new(6357222834896114791),
    BFieldElement::new(7776871531164456679),
    BFieldElement::new(8264739868177574620),
    BFieldElement::new(12732288338686680125),
    BFieldElement::new(13022293791945187811),
    BFieldElement::new(17403057736098613442),
    BFieldElement::new(2871266924987061743),
    BFieldElement::new(13286707530570640459),
    BFieldElement::new(9229362695439112266),
    BFieldElement::new(815317759014579856),
    BFieldElement::new(7447771153889267897),
    BFieldElement::new(2209002535000750347),
    BFieldElement::new(3280506473249596174),
    BFieldElement::new(13756142018694965622),
    BFieldElement::new(10518080861296830621),
    BFieldElement::new(16578355848983066277),
    BFieldElement::new(12732532221704648123),
    BFieldElement::new(3426526797578099186),
    BFieldElement::new(8563516248221808333),
    BFieldElement::new(13079317959606236131),
    BFieldElement::new(15645458946300428515),
    BFieldElement::new(9958819147895829140),
    BFieldElement::new(13028053188247480206),
    BFieldElement::new(6789511720078828478),
    BFieldElement::new(6583246594815170294),
    BFieldElement::new(4423695887326249884),
    BFieldElement::new(9751139665897711642),
    BFieldElement::new(10039202025292797758),
    BFieldElement::new(12208726994829996150),
    BFieldElement::new(6238795140281096003),
    BFieldElement::new(9113696057226188857),
    BFieldElement::new(9898705245385052191),
    BFieldElement::new(4213712701625520075),
    BFieldElement::new(8038355032286280912),
    BFieldElement::new(426685147605824917),
    BFieldElement::new(7673465577918025498),
    BFieldElement::new(8452867379070564008),
    BFieldElement::new(10827610229277395180),
    BFieldElement::new(16155539332955658546),
    BFieldElement::new(1575428636717115288),
    BFieldElement::new(8765972548498757598),
    BFieldElement::new(8405996249707890526),
    BFieldElement::new(14855028677418679455),
    BFieldElement::new(17878170012428694685),
    BFieldElement::new(16572621079016066883),
    BFieldElement::new(5311046098447994501),
    BFieldElement::new(10635376800783355348),
    BFieldElement::new(14205668690430323921),
    BFieldElement::new(1181422971831412672),
    BFieldElement::new(4651053123208915543),
    BFieldElement::new(12465667489477238576),
    BFieldElement::new(7300129031676503132),
    BFieldElement::new(13458544786180633209),
    BFieldElement::new(8946801771555977477),
    BFieldElement::new(14203890406114400141),
    BFieldElement::new(8219081892380458635),
    BFieldElement::new(6035067543134909245),
    BFieldElement::new(15140374581570897616),
    BFieldElement::new(4514006299509426029),
    BFieldElement::new(16757530089801321524),
    BFieldElement::new(13202061911440346802),
    BFieldElement::new(11227558237427129334),
    BFieldElement::new(315998614524336401),
    BFieldElement::new(11280705904396606227),
    BFieldElement::new(5798516367202621128),
    BFieldElement::new(17154761698338453414),
    BFieldElement::new(13574436947400004837),
    BFieldElement::new(3126509266905053998),
    BFieldElement::new(10740979484255925394),
    BFieldElement::new(9273322683773825324),
    BFieldElement::new(15349096509718845737),
    BFieldElement::new(14694022445619674948),
    BFieldElement::new(8733857890739087596),
    BFieldElement::new(3198488337424282101),
    BFieldElement::new(9521016570828679381),
    BFieldElement::new(11267736037298472148),
    BFieldElement::new(14825280481028844943),
    BFieldElement::new(1326588754335738002),
    BFieldElement::new(6200834522767914499),
    BFieldElement::new(1070210996042416038),
    BFieldElement::new(9140190343656907671),
    BFieldElement::new(15531381283521001952),
    BFieldElement::new(253143295675927354),
    BFieldElement::new(11977331414401291539),
    BFieldElement::new(13941376566367813256),
    BFieldElement::new(469904915148256197),
    BFieldElement::new(10873951860155749104),
    BFieldElement::new(3939719938926157877),
    BFieldElement::new(2271392376641547055),
    BFieldElement::new(4725974756185387075),
    BFieldElement::new(14827835543640648161),
    BFieldElement::new(17663273767033351157),
    BFieldElement::new(12440960700789890843),
    BFieldElement::new(16589620022628590428),
    BFieldElement::new(12838889473653138505),
    BFieldElement::new(11170336581460183657),
    BFieldElement::new(7583333056198317221),
    BFieldElement::new(6006908286410425140),
    BFieldElement::new(15648567098514276013),
    BFieldElement::new(188901633101859949),
    BFieldElement::new(12256163716419861419),
    BFieldElement::new(17319784688409668747),
    BFieldElement::new(9648971065289440425),
    BFieldElement::new(11370683735445551679),
    BFieldElement::new(11265203235776280908),
    BFieldElement::new(1737672785338087677),
    BFieldElement::new(5225587291780939578),
    BFieldElement::new(4739055740469849012),
    BFieldElement::new(1212344601223444182),
    BFieldElement::new(12958616893209019599),
    BFieldElement::new(7922060480554370635),
    BFieldElement::new(14661420107595710445),
    BFieldElement::new(11744359917257111592),
    BFieldElement::new(9674559564931202709),
    BFieldElement::new(8326110231976411065),
    BFieldElement::new(16856751238353701757),
    BFieldElement::new(7515652322254196544),
    BFieldElement::new(2062531989536141174),
    BFieldElement::new(3875321171362100965),
    BFieldElement::new(1164854003752487518),
    BFieldElement::new(3997098993859160292),
    BFieldElement::new(4074090397542250057),
    BFieldElement::new(3050858158567944540),
    BFieldElement::new(4568245569065883863),
    BFieldElement::new(14559440781022773799),
    BFieldElement::new(5401845794552358815),
    BFieldElement::new(6544584366002554176),
    BFieldElement::new(2511522072283652847),
    BFieldElement::new(9759884967674698659),
    BFieldElement::new(16411672358681189856),
    BFieldElement::new(11392578809073737776),
    BFieldElement::new(8013631514034873271),
    BFieldElement::new(11439549174997471674),
    BFieldElement::new(6373021446442411366),
    BFieldElement::new(12491600135569477757),
    BFieldElement::new(1017093281401495736),
    BFieldElement::new(663547836518863091),
    BFieldElement::new(16157302719777897692),
    BFieldElement::new(11208801522915446640),
    BFieldElement::new(10058178191286215107),
    BFieldElement::new(5521712058210208094),
    BFieldElement::new(3611681474253815005),
    BFieldElement::new(4864578569041337696),
    BFieldElement::new(12270319000993569289),
    BFieldElement::new(7347066511426336318),
    BFieldElement::new(6696546239958933736),
    BFieldElement::new(3335469193383486908),
    BFieldElement::new(12719366334180058014),
    BFieldElement::new(14123019207894489639),
    BFieldElement::new(11418186023060178542),
    BFieldElement::new(2042199956854124583),
    BFieldElement::new(17539253100488345226),
    BFieldElement::new(16240833881391672847),
    BFieldElement::new(11712520063241304909),
    BFieldElement::new(6456900719511754234),
    BFieldElement::new(1819022137223501306),
    BFieldElement::new(7371152900053879920),
    BFieldElement::new(6521878675261223812),
    BFieldElement::new(2050999666988944811),
    BFieldElement::new(8262038465464898064),
    BFieldElement::new(13303819303390508091),
    BFieldElement::new(12657292926928303663),
    BFieldElement::new(8794128680724662595),
    BFieldElement::new(4068577832515945116),
    BFieldElement::new(758247715040138478),
    BFieldElement::new(5600369601992438532),
    BFieldElement::new(3369463178350382224),
    BFieldElement::new(13763645328734311418),
    BFieldElement::new(9685701761982837416),
    BFieldElement::new(2711119809520557835),
    BFieldElement::new(11680482056777716424),
    BFieldElement::new(10958223503056770518),
    BFieldElement::new(4168390070510137163),
    BFieldElement::new(10823375744683484459),
    BFieldElement::new(5613197991565754677),
    BFieldElement::new(11781942063118564684),
    BFieldElement::new(9352512500813609723),
    BFieldElement::new(15997830646514778986),
    BFieldElement::new(7407352006524266457),
    BFieldElement::new(15312663387608602775),
    BFieldElement::new(3026364159907661789),
    BFieldElement::new(5698531403379362946),
    BFieldElement::new(2544271242593770624),
    BFieldElement::new(13104502948897878458),
    BFieldElement::new(7840062700088318710),
    BFieldElement::new(6028743588538970215),
    BFieldElement::new(6144415809411296980),
    BFieldElement::new(468368941216390216),
    BFieldElement::new(3638618405705274008),
    BFieldElement::new(11105401941482704573),
    BFieldElement::new(1850274872877725129),
    BFieldElement::new(1011155312563349004),
    BFieldElement::new(3234620948537841909),
    BFieldElement::new(3818372677739507813),
    BFieldElement::new(4863130691592118581),
    BFieldElement::new(8942166964590283171),
    BFieldElement::new(3639677194051371072),
    BFieldElement::new(15477372418124081864),
    BFieldElement::new(10322228711752830209),
    BFieldElement::new(9139111778956611066),
    BFieldElement::new(202171733050704358),
    BFieldElement::new(11982413146686512577),
    BFieldElement::new(11001000478006340870),
    BFieldElement::new(5491471715020327065),
    BFieldElement::new(6969114856449768266),
    BFieldElement::new(11088492421847219924),
    BFieldElement::new(12913509272810999025),
    BFieldElement::new(17366506887360149369),
    BFieldElement::new(7036328554328346102),
    BFieldElement::new(11139255730689011050),
    BFieldElement::new(2844974929907956457),
    BFieldElement::new(6488525141985913483),
    BFieldElement::new(2860098796699131680),
    BFieldElement::new(10366343151884073105),
    BFieldElement::new(844875652557703984),
    BFieldElement::new(1053177270393416978),
    BFieldElement::new(5189466196833763142),
    BFieldElement::new(1024738234713107670),
    BFieldElement::new(8846741799369572841),
    BFieldElement::new(14490406830213564822),
    BFieldElement::new(10577371742628912722),
    BFieldElement::new(3276210642025060502),
    BFieldElement::new(2605621719516949928),
    BFieldElement::new(5417148926702080639),
    BFieldElement::new(11100652475866543814),
    BFieldElement::new(5247366835775169839),
];

pub const MDS_INV: [BFieldElement; STATE_SIZE * STATE_SIZE] = [
    BFieldElement::new(1572742562154761373),
    BFieldElement::new(11904188991461183391),
    BFieldElement::new(16702037635100780588),
    BFieldElement::new(10395027733616703929),
    BFieldElement::new(8130016957979279389),
    BFieldElement::new(12091057987196709719),
    BFieldElement::new(14570460902390750822),
    BFieldElement::new(13452497170858892918),
    BFieldElement::new(7302470671584418296),
    BFieldElement::new(12930709087691977410),
    BFieldElement::new(6940810864055149191),
    BFieldElement::new(15479085069460687984),
    BFieldElement::new(15273989414499187903),
    BFieldElement::new(8742532579937987008),
    BFieldElement::new(78143684950290654),
    BFieldElement::new(10454925311792498315),
    BFieldElement::new(7789818152192856725),
    BFieldElement::new(3486011543032592030),
    BFieldElement::new(17188770042768805161),
    BFieldElement::new(10490412495468775616),
    BFieldElement::new(298640180115056798),
    BFieldElement::new(12895819509602002088),
    BFieldElement::new(1755013598313843104),
    BFieldElement::new(17242416429764373372),
    BFieldElement::new(993835663551930043),
    BFieldElement::new(17604339535769584753),
    BFieldElement::new(17954116481891390155),
    BFieldElement::new(332811330083846624),
    BFieldElement::new(14730023810555747819),
    BFieldElement::new(435413210797820565),
    BFieldElement::new(1781261080337413422),
    BFieldElement::new(4148505421656051973),
    BFieldElement::new(980199695323775177),
    BFieldElement::new(4706730905557535223),
    BFieldElement::new(12734714246714791746),
    BFieldElement::new(14273996233795959868),
    BFieldElement::new(7921735635146743134),
    BFieldElement::new(14772166129594741813),
    BFieldElement::new(2171393332099124215),
    BFieldElement::new(11431591906353698662),
    BFieldElement::new(1968460689143086961),
    BFieldElement::new(12435956952300281356),
    BFieldElement::new(18203712123938736914),
    BFieldElement::new(13226878153002754824),
    BFieldElement::new(4722189513468037980),
    BFieldElement::new(14552059159516237140),
    BFieldElement::new(2186026037853355566),
    BFieldElement::new(11286141841507813990),
    BFieldElement::new(565856028734827369),
    BFieldElement::new(13655906686104936396),
    BFieldElement::new(8559867348362880285),
    BFieldElement::new(2797343365604350633),
    BFieldElement::new(4465794635391355875),
    BFieldElement::new(10602340776590577912),
    BFieldElement::new(6532765362293732644),
    BFieldElement::new(9971594382705594993),
    BFieldElement::new(8246981798349136173),
    BFieldElement::new(4260734168634971109),
    BFieldElement::new(3096607081570771),
    BFieldElement::new(823237991393038853),
    BFieldElement::new(17532689952600815755),
    BFieldElement::new(12134755733102166916),
    BFieldElement::new(10570439735096051664),
    BFieldElement::new(18403803913856082900),
    BFieldElement::new(13128404168847275462),
    BFieldElement::new(16663835358650929116),
    BFieldElement::new(16546671721888068220),
    BFieldElement::new(4685011688485137218),
    BFieldElement::new(1959001578540316019),
    BFieldElement::new(16340711608595843821),
    BFieldElement::new(9460495021221259854),
    BFieldElement::new(3858517940845573321),
    BFieldElement::new(9427670160758976948),
    BFieldElement::new(18064975260450261693),
    BFieldElement::new(4905506444249847758),
    BFieldElement::new(15986418616213903133),
    BFieldElement::new(9282818778268010424),
    BFieldElement::new(9769107232941785010),
    BFieldElement::new(8521948467436343364),
    BFieldElement::new(7419602577337727529),
    BFieldElement::new(5926710664024036226),
    BFieldElement::new(11667040483862285999),
    BFieldElement::new(12291037072726747355),
    BFieldElement::new(12257844845576909578),
    BFieldElement::new(5216888292865522221),
    BFieldElement::new(4949589496388892504),
    BFieldElement::new(6571373688631618567),
    BFieldElement::new(10091372984903831417),
    BFieldElement::new(6240610640427541397),
    BFieldElement::new(6328690792776976228),
    BFieldElement::new(11836184983048970818),
    BFieldElement::new(12710419323566440454),
    BFieldElement::new(10374451385652807364),
    BFieldElement::new(8254232795575550118),
    BFieldElement::new(9866490979395302091),
    BFieldElement::new(12991014125893242232),
    BFieldElement::new(1063347186953727863),
    BFieldElement::new(2952135743830082310),
    BFieldElement::new(17315974856538709017),
    BFieldElement::new(14554512349953922358),
    BFieldElement::new(14134347382797855179),
    BFieldElement::new(17882046380988406016),
    BFieldElement::new(17463193400175360824),
    BFieldElement::new(3726957756828900632),
    BFieldElement::new(17604631050958608669),
    BFieldElement::new(7585987025945897953),
    BFieldElement::new(14470977033142357695),
    BFieldElement::new(10643295498661723800),
    BFieldElement::new(8871197056529643534),
    BFieldElement::new(8384208064507509379),
    BFieldElement::new(9280566467635869786),
    BFieldElement::new(87319369282683875),
    BFieldElement::new(1100172740622998121),
    BFieldElement::new(622721254307916221),
    BFieldElement::new(16843330035110191506),
    BFieldElement::new(13024130485811341782),
    BFieldElement::new(12334996107415540952),
    BFieldElement::new(461552745543935046),
    BFieldElement::new(8140793910765831499),
    BFieldElement::new(9008477689109468885),
    BFieldElement::new(17409910369122253035),
    BFieldElement::new(1804565454784197696),
    BFieldElement::new(5310948951638903141),
    BFieldElement::new(12531953612536647976),
    BFieldElement::new(6147853502869470889),
    BFieldElement::new(1125351356112285953),
    BFieldElement::new(6467901683012265601),
    BFieldElement::new(16792548587138841945),
    BFieldElement::new(14092833521360698433),
    BFieldElement::new(13651748079341829335),
    BFieldElement::new(10688258556205752814),
    BFieldElement::new(1823953496327460008),
    BFieldElement::new(2558053704584850519),
    BFieldElement::new(13269131806718310421),
    BFieldElement::new(4608410977522599149),
    BFieldElement::new(9221187654763620553),
    BFieldElement::new(4611978991500182874),
    BFieldElement::new(8855429001286425455),
    BFieldElement::new(5696709580182222832),
    BFieldElement::new(17579496245625003067),
    BFieldElement::new(5267934104348282564),
    BFieldElement::new(1835676094870249003),
    BFieldElement::new(3542280417783105151),
    BFieldElement::new(11824126253481498070),
    BFieldElement::new(9504622962336320170),
    BFieldElement::new(17887320494921151801),
    BFieldElement::new(6574518722274623914),
    BFieldElement::new(16658124633332643846),
    BFieldElement::new(13808019273382263890),
    BFieldElement::new(13092903038683672100),
    BFieldElement::new(501471167473345282),
    BFieldElement::new(11161560208140424921),
    BFieldElement::new(13001827442679699140),
    BFieldElement::new(14739684132127818993),
    BFieldElement::new(2868223407847949089),
    BFieldElement::new(1726410909424820290),
    BFieldElement::new(6794531346610991076),
    BFieldElement::new(6698331109000773276),
    BFieldElement::new(3680934785728193940),
    BFieldElement::new(8875468921351982841),
    BFieldElement::new(5477651765997654015),
    BFieldElement::new(12280771278642823764),
    BFieldElement::new(3619998794343148112),
    BFieldElement::new(6883119128428826230),
    BFieldElement::new(13512760119042878827),
    BFieldElement::new(3675597821767844913),
    BFieldElement::new(5414638790278102151),
    BFieldElement::new(3587251244316549755),
    BFieldElement::new(17100313981528550060),
    BFieldElement::new(11048426899172804713),
    BFieldElement::new(1396562484529002856),
    BFieldElement::new(2252873797267794672),
    BFieldElement::new(14201526079271439737),
    BFieldElement::new(16618356769072634008),
    BFieldElement::new(144564843743666734),
    BFieldElement::new(11912794688498369701),
    BFieldElement::new(10937102025343594422),
    BFieldElement::new(15432144252435329607),
    BFieldElement::new(2221546737981282133),
    BFieldElement::new(6015808993571140081),
    BFieldElement::new(7447996510907844453),
    BFieldElement::new(7039231904611782781),
    BFieldElement::new(2218118803134364409),
    BFieldElement::new(9472427559993341443),
    BFieldElement::new(11066826455107746221),
    BFieldElement::new(6223571389973384864),
    BFieldElement::new(13615228926415811268),
    BFieldElement::new(10241352486499609335),
    BFieldElement::new(12605380114102527595),
    BFieldElement::new(11403123666082872720),
    BFieldElement::new(9771232158486004346),
    BFieldElement::new(11862860570670038891),
    BFieldElement::new(10489319728736503343),
    BFieldElement::new(588166220336712628),
    BFieldElement::new(524399652036013851),
    BFieldElement::new(2215268375273320892),
    BFieldElement::new(1424724725807107497),
    BFieldElement::new(2223952838426612865),
    BFieldElement::new(1901666565705039600),
    BFieldElement::new(14666084855112001547),
    BFieldElement::new(16529527081633002035),
    BFieldElement::new(3475787534446449190),
    BFieldElement::new(17395838083455569055),
    BFieldElement::new(10036301139275236437),
    BFieldElement::new(5830062976180250577),
    BFieldElement::new(6201110308815839738),
    BFieldElement::new(3908827014617539568),
    BFieldElement::new(13269427316630307104),
    BFieldElement::new(1104974093011983663),
    BFieldElement::new(335137437077264843),
    BFieldElement::new(13411663683768112565),
    BFieldElement::new(7907493007733959147),
    BFieldElement::new(17240291213488173803),
    BFieldElement::new(6357405277112016289),
    BFieldElement::new(7875258449007392338),
    BFieldElement::new(16100900298327085499),
    BFieldElement::new(13542432207857463387),
    BFieldElement::new(9466802464896264825),
    BFieldElement::new(9221606791343926561),
    BFieldElement::new(10417300838622453849),
    BFieldElement::new(13201838829839066427),
    BFieldElement::new(9833345239958202067),
    BFieldElement::new(16688814355354359676),
    BFieldElement::new(13315432437333533951),
    BFieldElement::new(378443609734580293),
    BFieldElement::new(14654525144709164243),
    BFieldElement::new(1967217494445269914),
    BFieldElement::new(16045947041840686058),
    BFieldElement::new(18049263629128746044),
    BFieldElement::new(1957063364541610677),
    BFieldElement::new(16123386013589472221),
    BFieldElement::new(5923137592664329389),
    BFieldElement::new(12399617421793397670),
    BFieldElement::new(3403518680407886401),
    BFieldElement::new(6416516714555000604),
    BFieldElement::new(13286977196258324106),
    BFieldElement::new(17641011370212535641),
    BFieldElement::new(14823578540420219384),
    BFieldElement::new(11909888788340877523),
    BFieldElement::new(11040604022089158722),
    BFieldElement::new(14682783085930648838),
    BFieldElement::new(7896655986299558210),
    BFieldElement::new(9328642557612914244),
    BFieldElement::new(6213125364180629684),
    BFieldElement::new(16259136970573308007),
    BFieldElement::new(12025260496935037210),
    BFieldElement::new(1512031407150257270),
    BFieldElement::new(1295709332547428576),
    BFieldElement::new(13851880110872460625),
    BFieldElement::new(6734559515296147531),
    BFieldElement::new(17720805166223714561),
    BFieldElement::new(11264121550751120724),
    BFieldElement::new(7210341680607060660),
    BFieldElement::new(17759718475616004694),
    BFieldElement::new(610155440804635364),
    BFieldElement::new(3209025413915748371),
];

pub const ROUND_CONSTANTS: [BFieldElement; NUM_ROUNDS * STATE_SIZE * 2] = [
    BFieldElement::new(3006656781416918236),
    BFieldElement::new(4369161505641058227),
    BFieldElement::new(6684374425476535479),
    BFieldElement::new(15779820574306927140),
    BFieldElement::new(9604497860052635077),
    BFieldElement::new(6451419160553310210),
    BFieldElement::new(16926195364602274076),
    BFieldElement::new(6738541355147603274),
    BFieldElement::new(13653823767463659393),
    BFieldElement::new(16331310420018519380),
    BFieldElement::new(10921208506902903237),
    BFieldElement::new(5856388654420905056),
    BFieldElement::new(180518533287168595),
    BFieldElement::new(6394055120127805757),
    BFieldElement::new(4624620449883041133),
    BFieldElement::new(4245779370310492662),
    BFieldElement::new(11436753067664141475),
    BFieldElement::new(9565904130524743243),
    BFieldElement::new(1795462928700216574),
    BFieldElement::new(6069083569854718822),
    BFieldElement::new(16847768509740167846),
    BFieldElement::new(4958030292488314453),
    BFieldElement::new(6638656158077421079),
    BFieldElement::new(7387994719600814898),
    BFieldElement::new(1380138540257684527),
    BFieldElement::new(2756275326704598308),
    BFieldElement::new(6162254851582803897),
    BFieldElement::new(4357202747710082448),
    BFieldElement::new(12150731779910470904),
    BFieldElement::new(3121517886069239079),
    BFieldElement::new(14951334357190345445),
    BFieldElement::new(11174705360936334066),
    BFieldElement::new(17619090104023680035),
    BFieldElement::new(9879300494565649603),
    BFieldElement::new(6833140673689496042),
    BFieldElement::new(8026685634318089317),
    BFieldElement::new(6481786893261067369),
    BFieldElement::new(15148392398843394510),
    BFieldElement::new(11231860157121869734),
    BFieldElement::new(2645253741394956018),
    BFieldElement::new(15345701758979398253),
    BFieldElement::new(1715545688795694261),
    BFieldElement::new(3419893440622363282),
    BFieldElement::new(12314745080283886274),
    BFieldElement::new(16173382637268011204),
    BFieldElement::new(2012426895438224656),
    BFieldElement::new(6886681868854518019),
    BFieldElement::new(9323151312904004776),
    BFieldElement::new(14061124303940833928),
    BFieldElement::new(14720644192628944300),
    BFieldElement::new(3643016909963520634),
    BFieldElement::new(15164487940674916922),
    BFieldElement::new(18095609311840631082),
    BFieldElement::new(17450128049477479068),
    BFieldElement::new(13770238146408051799),
    BFieldElement::new(959547712344137104),
    BFieldElement::new(12896174981045071755),
    BFieldElement::new(15673600445734665670),
    BFieldElement::new(5421724936277706559),
    BFieldElement::new(15147580014608980436),
    BFieldElement::new(10475549030802107253),
    BFieldElement::new(9781768648599053415),
    BFieldElement::new(12208559126136453589),
    BFieldElement::new(14883846462224929329),
    BFieldElement::new(4104889747365723917),
    BFieldElement::new(748723978556009523),
    BFieldElement::new(1227256388689532469),
    BFieldElement::new(5479813539795083611),
    BFieldElement::new(8771502115864637772),
    BFieldElement::new(16732275956403307541),
    BFieldElement::new(4416407293527364014),
    BFieldElement::new(828170020209737786),
    BFieldElement::new(12657110237330569793),
    BFieldElement::new(6054985640939410036),
    BFieldElement::new(4339925773473390539),
    BFieldElement::new(12523290846763939879),
    BFieldElement::new(6515670251745069817),
    BFieldElement::new(3304839395869669984),
    BFieldElement::new(13139364704983394567),
    BFieldElement::new(7310284340158351735),
    BFieldElement::new(10864373318031796808),
    BFieldElement::new(17752126773383161797),
    BFieldElement::new(1934077736434853411),
    BFieldElement::new(12181011551355087129),
    BFieldElement::new(16512655861290250275),
    BFieldElement::new(17788869165454339633),
    BFieldElement::new(12226346139665475316),
    BFieldElement::new(521307319751404755),
    BFieldElement::new(18194723210928015140),
    BFieldElement::new(11017703779172233841),
    BFieldElement::new(15109417014344088693),
    BFieldElement::new(16118100307150379696),
    BFieldElement::new(16104548432406078622),
    BFieldElement::new(10637262801060241057),
    BFieldElement::new(10146828954247700859),
    BFieldElement::new(14927431817078997000),
    BFieldElement::new(8849391379213793752),
    BFieldElement::new(14873391436448856814),
    BFieldElement::new(15301636286727658488),
    BFieldElement::new(14600930856978269524),
    BFieldElement::new(14900320206081752612),
    BFieldElement::new(9439125422122803926),
    BFieldElement::new(17731778886181971775),
    BFieldElement::new(11364016993846997841),
    BFieldElement::new(11610707911054206249),
    BFieldElement::new(16438527050768899002),
    BFieldElement::new(1230592087960588528),
    BFieldElement::new(11390503834342845303),
    BFieldElement::new(10608561066917009324),
    BFieldElement::new(5454068995870010477),
    BFieldElement::new(13783920070953012756),
    BFieldElement::new(10807833173700567220),
    BFieldElement::new(8597517374132535250),
    BFieldElement::new(17631206339728520236),
    BFieldElement::new(8083932512125088346),
    BFieldElement::new(10460229397140806011),
    BFieldElement::new(16904442127403184100),
    BFieldElement::new(15806582425540851960),
    BFieldElement::new(8002674967888750145),
    BFieldElement::new(7088508235236416142),
    BFieldElement::new(2774873684607752403),
    BFieldElement::new(11519427263507311324),
    BFieldElement::new(14949623981479468161),
    BFieldElement::new(18169367272402768616),
    BFieldElement::new(13279771425489376175),
    BFieldElement::new(3437101568566296039),
    BFieldElement::new(11820510872362664493),
    BFieldElement::new(13649520728248893918),
    BFieldElement::new(13432595021904865723),
    BFieldElement::new(12153175375751103391),
    BFieldElement::new(16459175915481931891),
    BFieldElement::new(14698099486055505377),
    BFieldElement::new(14962427686967561007),
    BFieldElement::new(10825731681832829214),
    BFieldElement::new(12562849212348892143),
    BFieldElement::new(18054851842681741827),
    BFieldElement::new(16866664833727482321),
    BFieldElement::new(10485994783891875256),
    BFieldElement::new(8074668712578030015),
    BFieldElement::new(7502837771635714611),
    BFieldElement::new(8326381174040960025),
    BFieldElement::new(1299216707593490898),
    BFieldElement::new(12092900834113479279),
    BFieldElement::new(10147133736028577997),
    BFieldElement::new(12103660182675227350),
    BFieldElement::new(16088613802080804964),
    BFieldElement::new(10323305955081440356),
    BFieldElement::new(12814564542614394316),
    BFieldElement::new(9653856919559060601),
    BFieldElement::new(10390420172371317530),
    BFieldElement::new(7831993942325060892),
    BFieldElement::new(9568326819852151217),
    BFieldElement::new(6299791178740935792),
    BFieldElement::new(12692828392357621723),
    BFieldElement::new(10331476541693143830),
    BFieldElement::new(3115340436782501075),
    BFieldElement::new(17456578083689713056),
    BFieldElement::new(12924575652913558388),
    BFieldElement::new(14365487216177868031),
    BFieldElement::new(7211834371191912632),
    BFieldElement::new(17610068359394967554),
    BFieldElement::new(646302646073569086),
    BFieldElement::new(12437378932700222679),
    BFieldElement::new(2758591586601041336),
    BFieldElement::new(10952396165876183059),
    BFieldElement::new(8827205511644136726),
    BFieldElement::new(17572216767879446421),
    BFieldElement::new(12516044823385174395),
    BFieldElement::new(6380048472179557105),
    BFieldElement::new(1959389938825200414),
    BFieldElement::new(257915527015303758),
    BFieldElement::new(4942451629986849727),
    BFieldElement::new(1698530521870297461),
    BFieldElement::new(1802136667015215029),
    BFieldElement::new(6353258543636931941),
    BFieldElement::new(13791525219506237119),
    BFieldElement::new(7093082295632492630),
    BFieldElement::new(15409842367405634814),
    BFieldElement::new(2090232819855225051),
    BFieldElement::new(13926160661036606054),
    BFieldElement::new(389467431021126699),
    BFieldElement::new(4736917413147385608),
    BFieldElement::new(6217341363393311211),
    BFieldElement::new(4366302820407593918),
    BFieldElement::new(12748238635329332117),
    BFieldElement::new(7671680179984682360),
    BFieldElement::new(17998193362025085453),
    BFieldElement::new(432899318054332645),
    BFieldElement::new(1973816396170253277),
    BFieldElement::new(607886411884636526),
    BFieldElement::new(15080416519109365682),
    BFieldElement::new(13607062276466651973),
    BFieldElement::new(2458254972975404730),
    BFieldElement::new(15323169029557757131),
    BFieldElement::new(10953434699543086460),
    BFieldElement::new(13995946730291266219),
    BFieldElement::new(12803971247555868632),
    BFieldElement::new(3974568790603251423),
    BFieldElement::new(10629169239281589943),
    BFieldElement::new(2058261494620094806),
    BFieldElement::new(15905212873859894286),
    BFieldElement::new(11221574225004694137),
    BFieldElement::new(15430295276730781380),
    BFieldElement::new(10448646831319611878),
    BFieldElement::new(7559293484620816204),
    BFieldElement::new(15679753002507105741),
    BFieldElement::new(6043747003590355195),
    BFieldElement::new(3404573815097301491),
    BFieldElement::new(13392826344874185313),
    BFieldElement::new(6464466389567159772),
    BFieldElement::new(8932733991045074013),
    BFieldElement::new(6565970376680631168),
    BFieldElement::new(7050411859293315754),
    BFieldElement::new(9763347751680159247),
    BFieldElement::new(3140014248604700259),
    BFieldElement::new(5621238883761074228),
    BFieldElement::new(12664766603293629079),
    BFieldElement::new(6533276137502482405),
    BFieldElement::new(914829860407409680),
    BFieldElement::new(14599697497440353734),
    BFieldElement::new(16400390478099648992),
    BFieldElement::new(1619185634767959932),
    BFieldElement::new(16420198681440130663),
    BFieldElement::new(1331388886719756999),
    BFieldElement::new(1430143015191336857),
    BFieldElement::new(14618841684410509097),
    BFieldElement::new(1870494251298489312),
    BFieldElement::new(3783117677312763499),
    BFieldElement::new(16164771504475705474),
    BFieldElement::new(6996935044500625689),
    BFieldElement::new(4356994160244918010),
    BFieldElement::new(13579982029281680908),
    BFieldElement::new(8835524728424198741),
    BFieldElement::new(13281017722683773148),
    BFieldElement::new(2669924686363521592),
    BFieldElement::new(15020410046647566094),
    BFieldElement::new(9534143832529454683),
    BFieldElement::new(156263138519279564),
    BFieldElement::new(17421879327900831752),
    BFieldElement::new(9524879102847422379),
    BFieldElement::new(5120021146470638642),
    BFieldElement::new(9588770058331935449),
    BFieldElement::new(1501841070476096181),
    BFieldElement::new(5687728871183511192),
    BFieldElement::new(16091855309800405887),
    BFieldElement::new(17307425956518746505),
    BFieldElement::new(1162636238106302518),
    BFieldElement::new(8756478993690213481),
    BFieldElement::new(6898084027896327288),
    BFieldElement::new(8485261637658061794),
    BFieldElement::new(4169208979833913382),
    BFieldElement::new(7776158701576840241),
    BFieldElement::new(13861841831073878156),
    BFieldElement::new(4896983281306117497),
    BFieldElement::new(6056805506026814259),
    BFieldElement::new(15706891000994288769),
];

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RescuePrimeRegularState {
    pub state: [BFieldElement; STATE_SIZE],
}

impl RescuePrimeRegularState {
    #[inline]
    const fn new() -> RescuePrimeRegularState {
        RescuePrimeRegularState {
            state: [BFIELD_ZERO; STATE_SIZE],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct RescuePrimeRegular {}

impl RescuePrimeRegular {
    #[inline]
    fn batch_square(array: &mut [BFieldElement; STATE_SIZE]) {
        for a in array.iter_mut() {
            *a = a.square();
        }
    }

    #[inline]
    fn batch_square_n<const N: usize>(array: &mut [BFieldElement; STATE_SIZE]) {
        for _ in 0..N {
            Self::batch_square(array);
        }
    }

    #[inline]
    fn batch_mul_into(
        array: &mut [BFieldElement; STATE_SIZE],
        operand: [BFieldElement; STATE_SIZE],
    ) {
        for (a, b) in array.iter_mut().zip_eq(operand.iter()) {
            *a *= *b;
        }
    }

    #[inline]
    fn batch_mod_pow_alpha_inv(array: [BFieldElement; STATE_SIZE]) -> [BFieldElement; STATE_SIZE] {
        // alpha^-1 = 0b1001001001001001001001001001000110110110110110110110110110110111

        // credit to Winterfell for this decomposition into 72 multiplications
        // (1) base^10
        // (2) base^100 = (1) << 1
        // (3) base^100100 = (2) << 3 + (2)
        // (4) base^100100100100 = (3) << 6 + (3)
        // (5) base^100100100100100100100100 = (4) << 12 + (4)
        // (6) base^100100100100100100100100100100 = (5) << 6 + (3)
        // (7) base^1001001001001001001001001001000100100100100100100100100100100
        //     = (6) << 31 + (6)
        // (r) base^1001001001001001001001001001000110110110110110110110110110110111
        //     = ((7) << 1 + (6)) << 2 + (2)  +  (1)  +  1

        let mut p1 = array;
        Self::batch_square(&mut p1);

        let mut p2 = p1;
        Self::batch_square(&mut p2);

        let mut p3 = p2;
        Self::batch_square_n::<3>(&mut p3);
        Self::batch_mul_into(&mut p3, p2);

        let mut p4 = p3;
        Self::batch_square_n::<6>(&mut p4);
        Self::batch_mul_into(&mut p4, p3);

        let mut p5 = p4;
        Self::batch_square_n::<12>(&mut p5);
        Self::batch_mul_into(&mut p5, p4);

        let mut p6 = p5;
        Self::batch_square_n::<6>(&mut p6);
        Self::batch_mul_into(&mut p6, p3);

        let mut p7 = p6;
        Self::batch_square_n::<31>(&mut p7);
        Self::batch_mul_into(&mut p7, p6);

        let mut result = p7;
        Self::batch_square(&mut result);
        Self::batch_mul_into(&mut result, p6);
        Self::batch_square_n::<2>(&mut result);
        Self::batch_mul_into(&mut result, p2);
        Self::batch_mul_into(&mut result, p1);
        Self::batch_mul_into(&mut result, array);
        result
    }

    #[inline]
    fn batch_mod_pow_alpha(array: [BFieldElement; STATE_SIZE]) -> [BFieldElement; STATE_SIZE] {
        let mut result = array;
        Self::batch_square(&mut result);
        Self::batch_mul_into(&mut result, array);
        Self::batch_square(&mut result);
        Self::batch_mul_into(&mut result, array);
        result
    }

    #[allow(dead_code)]
    fn batch_mod_pow(
        array: [BFieldElement; STATE_SIZE],
        power: u64,
    ) -> [BFieldElement; STATE_SIZE] {
        let mut acc = [BFieldElement::one(); STATE_SIZE];
        for i in (0..64).rev() {
            if i != 63 {
                Self::batch_square(&mut acc);
            }
            if power & (1 << i) != 0 {
                Self::batch_mul_into(&mut acc, array);
            }
        }

        acc
    }

    /// xlix_round
    /// Apply one round of the XLIX permutation.
    #[inline]
    fn xlix_round(sponge: &mut RescuePrimeRegularState, round_index: usize) {
        debug_assert!(
            round_index < NUM_ROUNDS,
            "Cannot apply {}th round; only have {} in total.",
            round_index,
            NUM_ROUNDS
        );

        // S-box
        // for i in 0..STATE_SIZE {
        //     self.state[i] = self.state[i].mod_pow_u64(ALPHA);
        // }
        //
        sponge.state = Self::batch_mod_pow_alpha(sponge.state);

        // MDS matrix
        let mut v: [BFieldElement; STATE_SIZE] = [BFIELD_ZERO; STATE_SIZE];
        for i in 0..STATE_SIZE {
            for j in 0..STATE_SIZE {
                v[i] += MDS[i * STATE_SIZE + j] * sponge.state[j];
            }
        }
        sponge.state = v;

        // round constants A
        for i in 0..STATE_SIZE {
            sponge.state[i] += ROUND_CONSTANTS[round_index * STATE_SIZE * 2 + i];
        }

        // Inverse S-box
        // for i in 0..STATE_SIZE {
        //     self.state[i] = self.state[i].mod_pow_u64(ALPHA_INV);
        // }
        //
        // self.state = Self::batch_mod_pow(self.state, ALPHA_INV);
        sponge.state = Self::batch_mod_pow_alpha_inv(sponge.state);

        // MDS matrix
        for i in 0..STATE_SIZE {
            v[i] = BFIELD_ZERO;
            for j in 0..STATE_SIZE {
                v[i] += MDS[i * STATE_SIZE + j] * sponge.state[j];
            }
        }
        sponge.state = v;

        // round constants B
        for i in 0..STATE_SIZE {
            sponge.state[i] += ROUND_CONSTANTS[round_index * STATE_SIZE * 2 + STATE_SIZE + i];
        }
    }

    /// xlix
    /// XLIX is the permutation defined by Rescue-Prime. This
    /// function applies XLIX to the state of a sponge.
    #[inline]
    fn xlix(sponge: &mut RescuePrimeRegularState) {
        for i in 0..NUM_ROUNDS {
            Self::xlix_round(sponge, i);
        }
    }

    /// hash_10
    /// Hash 10 elements, or two digests. There is no padding because
    /// the input length is fixed.
    pub fn hash_10(input: &[BFieldElement; 10]) -> [BFieldElement; 5] {
        let mut sponge = RescuePrimeRegularState::new();

        // absorb once
        sponge.state[..10].copy_from_slice(input);

        // apply domain separation for fixed-length input
        sponge.state[10] = BFIELD_ONE;

        // apply xlix
        Self::xlix(&mut sponge);

        // squeeze once
        sponge.state[..5].try_into().unwrap()
    }

    /// hash_varlen hashes an arbitrary number of field elements.
    ///
    /// Takes care of padding by applying the padding rule: append a single 1 ∈ Fp
    /// and as many 0 ∈ Fp elements as required to make the number of input elements
    /// a multiple of `RATE`.
    pub fn hash_varlen(input: &[BFieldElement]) -> [BFieldElement; 5] {
        let mut sponge = RescuePrimeRegularState::new();

        // pad input
        let mut padded_input = input.to_vec();
        padded_input.push(BFIELD_ONE);
        while padded_input.len() % RATE != 0 {
            padded_input.push(BFIELD_ZERO);
        }

        // absorb
        while !padded_input.is_empty() {
            for (sponge_state_element, &input_element) in sponge
                .state
                .iter_mut()
                .take(RATE)
                .zip_eq(padded_input.iter().take(RATE))
            {
                *sponge_state_element += input_element;
            }
            padded_input.drain(..RATE);
            Self::xlix(&mut sponge);
        }

        // squeeze once
        sponge.state[..5].try_into().unwrap()
    }

    /// trace
    /// Produces the execution trace for one invocation of XLIX
    pub fn trace(input: &[BFieldElement; 10]) -> [[BFieldElement; STATE_SIZE]; 1 + NUM_ROUNDS] {
        let mut trace = [[BFIELD_ZERO; STATE_SIZE]; 1 + NUM_ROUNDS];
        let mut sponge = RescuePrimeRegularState::new();

        // absorb
        sponge.state[0..RATE].copy_from_slice(input);

        // domain separation
        sponge.state[RATE] = BFIELD_ONE;

        // record trace
        trace[0] = sponge.state;

        // apply N rounds
        for round_index in 0..NUM_ROUNDS {
            // apply round function to state
            Self::xlix_round(&mut sponge, round_index);

            // record trace
            trace[1 + round_index] = sponge.state;
        }

        trace
    }

    /// full_trace
    /// Produces the execution trace for one invocation of XLIX
    pub fn full_trace(
        state: [BFieldElement; STATE_SIZE],
    ) -> [[BFieldElement; STATE_SIZE]; 1 + NUM_ROUNDS] {
        let mut trace = [[BFIELD_ZERO; STATE_SIZE]; 1 + NUM_ROUNDS];
        let mut state = RescuePrimeRegularState { state };

        // record trace
        trace[0] = state.state;

        // apply N rounds
        for round_index in 0..NUM_ROUNDS {
            // apply round function to state
            Self::xlix_round(&mut state, round_index);

            // record trace
            trace[1 + round_index] = state.state;
        }

        trace
    }
}

impl AlgebraicHasher for RescuePrimeRegular {
    fn hash_slice(elements: &[BFieldElement]) -> Digest {
        Digest::new(RescuePrimeRegular::hash_varlen(elements))
    }

    fn hash_pair(left: &Digest, right: &Digest) -> Digest {
        let mut input = [BFIELD_ZERO; 10];
        input[..DIGEST_LENGTH].copy_from_slice(&left.values());
        input[DIGEST_LENGTH..].copy_from_slice(&right.values());
        Digest::new(RescuePrimeRegular::hash_10(&input))
    }
}

#[cfg(test)]
mod rescue_prime_regular_tests {
    use itertools::Itertools;
    use num_traits::Zero;

    use super::*;

    #[test]
    fn test_compliance() {
        // hash 10, first batch
        let targets_first_batch: [[u64; 5]; 10] = [
            [
                13772361690486541727,
                13930107128811878860,
                10372771888229343819,
                3089663060285256636,
                13524175591068575265,
            ],
            [
                6189874031558670378,
                10677984284721664052,
                11094778135221566221,
                14014056563831415241,
                17624827189757302581,
            ],
            [
                8051330103982111662,
                11028112094367299718,
                13272421194765666592,
                10549698261930941220,
                11916714144939538031,
            ],
            [
                16599150548794199309,
                10493776575292205727,
                1383032563775085972,
                13461691265189540514,
                5850245107934240031,
            ],
            [
                1945888927084124766,
                1359531585423726517,
                571115580035577760,
                11111259412951139324,
                5196253525869594875,
            ],
            [
                7114248945826389783,
                10711114340774706221,
                1725038494551646750,
                5126322762295353904,
                263075078948948040,
            ],
            [
                6920355036596643958,
                1429967828412209302,
                17849302350128548438,
                17338513521733790109,
                16932869691567724957,
            ],
            [
                105964463927719565,
                9782095356206296143,
                8894170595070814364,
                5355551316895236465,
                5948538211133255,
            ],
            [
                17155306598521508982,
                14472525374811419229,
                16093490088522805438,
                3181770408060478711,
                1640164811138798343,
            ],
            [
                4347619024993392377,
                3485116829811114387,
                4909926852610789363,
                7049913000384061828,
                5525683604796387635,
            ],
        ];

        for (i, target) in targets_first_batch.into_iter().enumerate() {
            let expected = target.map(BFieldElement::new);
            let mut input = [BFieldElement::zero(); 10];
            input[input.len() - 1] = BFieldElement::from(i as u64);
            let actual = RescuePrimeRegular::hash_10(&input);
            assert_eq!(expected, actual);
        }

        // hash 10, second batch
        let targets_second_batch: [[u64; 5]; 10] = [
            [
                1161630535021658359,
                12802704886281901257,
                5310064968286039002,
                16402665326204561132,
                15530842331109449708,
            ],
            [
                14361074708917363055,
                2973268553816013426,
                14423563135619089510,
                12919806918032553606,
                1236538015722961393,
            ],
            [
                15557076570140697692,
                18184365344427218921,
                13484852599853983709,
                13107035934236521262,
                12263607846531896566,
            ],
            [
                6606345110125563613,
                4726659718853731050,
                10369781804093423850,
                11406738559336087061,
                9592587996153825778,
            ],
            [
                6941529093020236364,
                10640545875110113888,
                11208725571809152547,
                11694492597767881733,
                16029873915245617377,
            ],
            [
                1593003208162720544,
                5732973444506631828,
                12557012035524685001,
                11271950808101345806,
                9333507744968394366,
            ],
            [
                10813877996011332235,
                5998166913909153813,
                629292377717800414,
                15727071196577879481,
                4491609621508834803,
            ],
            [
                4266637802759423528,
                13789207577808039094,
                10064576707545119428,
                7511635467227872433,
                6960370489820239537,
            ],
            [
                14498937510007389283,
                14239237126868820990,
                11153197751373890408,
                4310772621580754358,
                2864799328000030300,
            ],
            [
                6189874031558670378,
                10677984284721664052,
                11094778135221566221,
                14014056563831415241,
                17624827189757302581,
            ],
        ];
        for (i, target) in targets_second_batch.into_iter().enumerate() {
            let expected = target.map(BFieldElement::new);
            let mut input = [BFieldElement::zero(); 10];
            input[i] = BFieldElement::one();
            let actual = RescuePrimeRegular::hash_10(&input);
            assert_eq!(expected, actual);
        }

        // hash varlen, third batch
        let targets_third_batch: [[u64; 5]; 20] = [
            [
                9954340196098770044,
                16766478858550921719,
                14795358262939961687,
                5971715312175262159,
                10621735453321362721,
            ],
            [
                12391337510008436236,
                15657559504547420941,
                9377428313701566093,
                6455690240973939776,
                17925569643122616714,
            ],
            [
                14112140563534526715,
                12091338198119135732,
                16277751626976027823,
                4331384491863420413,
                15800084865512048249,
            ],
            [
                15410161320350300674,
                12862508375582878113,
                1871289024748006724,
                1120358582983879653,
                10608258519034552134,
            ],
            [
                1107344292722494872,
                17391364595468230070,
                4218215235563531160,
                7497689442794338714,
                3900922406630849053,
            ],
            [
                16822019589580184555,
                7989891526544888053,
                14569641731101827620,
                17919386380356552805,
                7463713352054333042,
            ],
            [
                674513153759557192,
                5885835165060751739,
                7545202825089012468,
                7455443267898983077,
                11460188487338022037,
            ],
            [
                1452905481212349824,
                4602871015292638258,
                16799505315703203495,
                15502476305285227202,
                14418240163510509007,
            ],
            [
                2169799351531593851,
                141901148303731658,
                12571509576917037512,
                2730951366471393395,
                10868840823954592153,
            ],
            [
                5004294054773159410,
                15035327361975356310,
                14190623520133446702,
                16843665251688123638,
                4543333205754908370,
            ],
            [
                16497508931324347828,
                10379016827033660777,
                5027352471010305075,
                15362732119758725484,
                13390969807239861733,
            ],
            [
                1743559568736995800,
                11815709956493259346,
                5763576938286686837,
                7541138447063081288,
                17969015713376415699,
            ],
            [
                10441678943133242957,
                15290592304889070108,
                18288160234755515065,
                3671382450876247307,
                3447450231474938402,
            ],
            [
                11057569330409963321,
                4984952761946312859,
                16529019269578375042,
                1908152979369527531,
                7121827819059879337,
            ],
            [
                17067972955397432517,
                2912062349216497629,
                15263972887304976204,
                9246522127607732383,
                17610927156233305697,
            ],
            [
                5980270367087450085,
                2990338491388854267,
                3198993023459349000,
                5035257001959372883,
                5260797048498744804,
            ],
            [
                8542899768037601505,
                5239516840302652488,
                2299137376555803866,
                952010414036958775,
                9717098700918296507,
            ],
            [
                8231024478155080292,
                9594681520895674398,
                191017068357133911,
                1512051294906340420,
                12055973608766483576,
            ],
            [
                16653142742451850722,
                9252945525340562222,
                4805241920959388929,
                937662086458078174,
                17775208482321191727,
            ],
            [
                14634923894397095166,
                1247948061415695017,
                3048493836613607105,
                2432649783604354905,
                7424726151688166928,
            ],
        ];
        for (i, target) in targets_third_batch.into_iter().enumerate() {
            let expected = target.map(BFieldElement::new);
            let input = (0..i as u64).map(BFieldElement::new).collect_vec();
            let actual = RescuePrimeRegular::hash_varlen(&input);
            assert_eq!(expected, actual);
        }
    }
}
